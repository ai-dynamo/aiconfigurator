# SGLang 0.5.14 Hopper/Blackwell alignment ledger

This ledger records changes that can regress one platform while fixing the
other. Update it whenever SGLang 0.5.14 collector or shared consumer behavior
is changed for SM90, SM100/103, SM120, or SM89.

## Fixed contract

- Framework: stock SGLang `0.5.14` only.
- Runtime: `lmsysorg/sglang:v0.5.14-cu130`, digest
  `sha256:5027e95bf6ec536856b1b52a91d1f35ff5c564ab83e8a94758a169ff09bb8df3`.
- Base under test: `9a5fe012`, whose direct parent is the GLM-5.2 change
  `f4d1bcc7`.
- Last separately validated SM90 line: `9177387b`; its relevant SGLang work is
  represented by `c8c00f42`.
- These are divergent lines, not a linear history. Their merge base is
  `66c6e05f`: one side ends at `c8c00f42 -> 9177387b`, while the current side
  ends at `f4d1bcc7 -> 9a5fe012`. A contract present on the old line and absent
  on the current line therefore has a first-known-missing state, not
  necessarily one identifiable dropping commit.
- Platform priority: SM90 and SM100, then SM120, then SM89. Keep SM103
  distinct where the framework does.
- Current SM90 execution host is an NVIDIA H20-3e. `h200_sxm` is used only as
  the existing SM90 collector/system selector; H20 measurements must not be
  published as H200 data.
- Blackwell conclusions remain source-derived until rerun on the matching
  hardware.

## Current evidence bundle

- Host-local run record:
  `/raid/aic_sglang_0514_sm90_post_b200_20260701/RUN_RECORD.md`.
- Frozen source/checkpoint/output root:
  `/raid/aic_sglang_0514_sm90_post_b200_20260701/full`.
- Frozen snapshot manifest SHA256:
  `cdfca0b90034e858b8445b18156977026281a7be1a7e1ca3ab78802aeb34ed73`.
- Frozen run's tracked working-diff SHA256 (before later documentation-only
  ledger updates):
  `2381af1392c7b7f40318fd533c97dd85d81de1beba354f05b9212c3c1526fa9e`.
- The second read-only source snapshot for post-DSA work through the strict GLM
  comparator is
  `/raid/aic_sglang_0514_sm90_post_b200_20260701/post_freeze/code_snapshot`.
  It remains based on `9a5fe012`; its 1,488-file manifest SHA256 is
  `3f58b147ae3fea83cb6897be6bafd93a3cd0e1bb856ccdd03481dfbde6005ecd`,
  its complete tracked binary patch SHA256 is
  `d8cc98aa6af403402482cff99225605b256a24a6f26176000203441fc945582e`,
  and its seven-file untracked-content manifest SHA256 is
  `544d29669a13883f7fc8a237f5fec4b789b9a0ff9288da16e8b049b87cd65eab`.
  This snapshot includes the DSV4 phase correction, strict failure accounting,
  mHC teardown, MoE consolidation, DSA decision records, and test pruning.
  It does not alter or relabel the original `cdfca0b9...` snapshot. Any later
  execution-code change requires a new snapshot identity before more GPU data
  is accepted; documentation-only ledger appendices do not mutate the frozen
  source used by the run.
- The row-complete but semantically rejected GLM top-k ragged snapshot is
  `/raid/aic_sglang_0514_sm90_post_b200_20260701/post_glm_topk_chunk_v1/code_snapshot`,
  manifest SHA256
  `893724083c1f11a0944cc87d97b98ce5c604f624b1ada2661f75fd310a1658bc`.
  Its fresh top-k artifact is under `remaining/glm_topk_chunk_v1`; the
  split-provenance three-op bundle is under `remaining/glm_sparse_composite_v1`
  and its `ARTIFACTS.sha256` file has SHA256 `5fc2ab60...`.
- Subsequent corrections exist only as mutable worktree checkpoints. GLM
  `c5cba92d...` fixed context but retained present-zero decode `row_starts`;
  `8f05cbfd...` omitted it but retained pad4 decode stride; `6da3683b...`
  corrected stride but did not explicitly release each materialized chunk page
  table before allocating the next pair. All are superseded partial candidates.
  At the current pre-GPU checkpoint, GLM SHA256 is `1194e6b6...`, DSV3.2
  prose-only wrapper SHA256 is `383e2a59...`, and focused-test SHA256 is
  `89026f7d...`; 37/37 focused tests and Ruff/format/diff checks pass. None is a
  frozen execution snapshot and none has a GPU artifact. The append-only
  selector/decode/lifetime chronology is recorded below.
- The run root is evidence, not a tracked delivery artifact. Durable counts,
  failure tuples, provenance, and reverse-platform requirements must be copied
  into this ledger before handoff.
- Hardware status for this effort: the recorded SM90 stages execute on H20-3e.
  GLM top-k has a structurally complete 6,664-row ragged artifact, but the
  completed selector audit rejects its context backend, and the follow-up
  exact-kernel audit rejects all 74 nonzero decode rows because they selected
  the prefill leaf by passing an optional that production decode omits; 68 also
  used wrong physical stride. The other 214 rows return zero before launch. A
  source-aligned paged worktree patch still needs a new snapshot, GPU value and
  leaf-kernel smoke, and complete rerun;
  DSV4 bounded-pool and MoE-last also remain separate work. SM100, SM103, SM120, and SM89 are not
  hardware-validated here.
  Existing branches and comments for those SMs are source-derived or inherited
  until a matching node reruns them.

## Cross-platform provenance

| ID | Contract and provenance | SM90 evidence | Blackwell continuation |
| --- | --- | --- | --- |
| `ATTN-VARIANT-KEY-0514` | Dense-attention `(kernel_source, attention_chunk_size, v_head_dim)` keys first appeared in `c8c00f42`; its parent `01aa8fd9` is last-good for untouched vLLM/TRT-LLM consumers. The same `c8c00f42` change both added the intended SGLang 0.5.14 keys and incorrectly made shared consumers source-sensitive. Replaying the old-line work onto `9a5fe012` reproduced that first-bad behavior; neither `f4d1bcc7` nor clean `9a5fe012` introduced it. The correction is new shared-consumer gating: only exact SGLang 0.5.14 keeps variants, all other framework/version rows normalize to `(None, 0, head_dim)`. | Full retained dense-attention H20/SM90 collection passed with zero failures: context 51,258, generation 40,468, encoder 7,679 unique keys. MiMo-V2 asymmetric V-head, Llama-4 chunk 8192, and Gemma-4 Triton signatures are complete. Packaged H200 SGLang 0.5.10 keys are strict subsets after legacy projection. Python/Rust reverse queries and Rust `embedded_round_trip` pass. | For SM100/103/120, probe context and decode selectors for every architecture with an explicit source mapping, including MiMo-V2, Llama-4, Gemma-4, GPT-OSS, MiMo, NemotronH, and Qwen3.5 variants. After any shared-loader/query edit, rerun the exact SM90 signatures plus packaged vLLM/TRT-LLM Python and Rust reverse queries. Do not infer other-framework source mappings from SGLang. |
| `ATTN-PER-MODEL-ROUTING` | B200 auto-healing `f027123f` added the broad `SM100 + head_dim=192 -> Triton` workaround and `31b6e31b` retained it with a TODO. Clean `9a5fe012` instead rejects or defers those profiles. The divergent `c8c00f42` line, replayed in the current uncommitted case-generator/collector changes, replaces that shape-only rule with per-model SGLang 0.5.14 routing for MiMo-V2, Gemma-4, GPT-OSS, Llama-4, NemotronH, and Qwen3.5, and keeps Kimi attention excluded from SGLang while vLLM still emits it. This is a third, source-specific contract rather than blindly restoring either earlier branch. A post-freeze correction in `collector/vllm/collect_attn*.py` and `collector/trtllm/collect_attn.py` now passes the caller backend explicitly so the shared framework filter also applies outside SGLang. | H20 full dense collection proves the emitted SGLang routes execute: MiMo-V2 SM90 uses FA3 with `(QK=192,V=128)`, while Gemma-4 uses Triton; focused getter tests prove MiMo SM100/103/120 selects Triton and Kimi emits zero SGLang but nonzero vLLM configs. The hardware run has zero missing dense keys. A focused AST contract proves both context/generation production callers identify `vllm`/`trtllm`; real untouched-framework plan counts remain pending. | Probe every listed model on B200; keep SM103 and SM120 separate. Re-run MiMo SM90 FA3 and Gemma/Kimi controls after any Blackwell change. Run real vLLM/TRT-LLM plan/count regressions in their pinned images before closing this row; helper and call-wiring tests alone do not prove their expanded plans. |
| `ENGINE-SPEC-V2-ATTENTION` | `c8c00f42` (parent `01aa8fd9`) changed Python/Rust EngineSpec from schema 1 to 2, added attention source/chunk/V-head fields, resolved `latest` to the actual database version, and threaded that resolved version into operation lookup. Current Rust deserialization deliberately rejects a schema-1 bincode rather than guessing defaults. This consumer boundary is distinct from collector exact-version support and must not be silently relaxed during a platform fix. | Python EngineSpec tests cover exact SGLang 0.5.14 variant fields, custom systems-path/latest resolution, legacy SGLang 0.5.10 historical keys, and untouched vLLM/TRT-LLM keys. Rust native attention and packaged-data round trips pass apart from separately recorded baseline debt. No existing serialized v1 artifact is claimed compatible. | On SM100/103/120, compile and deserialize a schema-2 engine against the resolved database version and execute representative attention queries. After any schema/default edit, rerun SM90 exact-variant, packaged 0.5.10, vLLM/TRT-LLM, and explicit schema-1 rejection controls. If v1 migration becomes a requirement, make it a separate reviewed wire migration rather than a collector workaround. |
| `WORKER-FATAL-RECYCLE` | `c8c00f42` first added typed CUDA-OOM/fatal detection, return-valued `EXIT_CODE_RESTART`, and `gc`/device-cache cleanup after nonfatal task errors; clean `9a5fe012` lacks that complete lifecycle. Sibling `812bac72` independently re-added only CUDA-OOM fatal detection, showing the same contract can be lost and rediscovered. Frozen snapshot `cdfca0b9...` contains the replayed `c8c00f42` lifecycle in `collector/collect.py`; current post-freeze changes have not altered it. | Focused worker tests cover CUDA OOM/fatal versus CPU OOM/nonfatal classification, returned restart signals, repeated restarts, and cleanup. In the H20 full DSA run, fatal illegal-access workers were replaced, later tasks continued, checkpoint failed IDs remained exact, and all task-owned GPU processes were gone after completion. Full GDN/MoE restart-path hardware validation remains pending. | This is cross-platform infrastructure, not a Hopper workaround. On B200 and later SM103/120 runs, force one fatal CUDA task followed by a known-good task and one ordinary exception followed by a known-good task; verify new-worker recovery, no inherited allocation, and exact done/failed checkpoint accounting. Exercise both a GDN-style returned restart and a MoE task before changing this row. Never catch a poisoned CUDA context merely to keep the same worker alive. |
| `GEMMA-GELU-ALIGN` | Gemma4 GELU metadata/filter exists on the divergent `c8c00f42 -> 9177387b` line but is first-known-missing on `f4d1bcc7 -> 9a5fe012`; there is no proven linear dropping commit. Restoring it is a lost-contract restoration, not speculative behavior. | Hopper vector width is 8. TP16/32 invalid local widths are pruned, changing retained Gemma rows from 3,078 to 2,349; targeted BF16 MoE smoke passed. | Keep SM100/103 vector width 16. Current code also applies width 16 to SM120 via `sm_version >= 100`; that is source-derived and hardware-unvalidated, so probe SM120 independently before calling it supported. Regenerate each SM count and smoke the first valid/invalid TP boundary. Every later Blackwell alignment edit must rerun the SM90 2,349-row getter contract. |
| `DSV4-ROUTER-FP32` | `9a5fe012` first introduced this framework-quantized DSV4 MoE collector path and sent synthetic logits without the FP32 contract used by production `linear_bf16_fp32`; there is no same-path last-good. The current cast is a new-path correction. | Synthetic balanced/power-law logits are FP32; DSV4-Pro W4A16 targeted smoke passed. | Rerun DSV4 Flash/Pro routing, top-k, and MoE on SM100/103 and separately inspect SM120. Preserve each platform's FP4-expert output/bypass contract, then rerun the SM90 DSV4-Pro W4A16 smoke. |
| `DSV4-W4A16-ALIGN` | `9a5fe012` first enabled SM90 DSV4-Pro W4A16 without the TP-local 128-element alignment required by `Mxfp4FlashinferCutlassMoEMethod`; no same-path last-good exists. | The predicate is SM90 W4A16 FP4-experts only. TP16/32 are pruned and retained cases change 450 to 300; targeted W4A16 smoke passed. | Do not apply this Hopper predicate to SM100/103 W4A8 generated paths or assume it for SM120. Regenerate and smoke TP1/2/4/8 on B200, and rerun the SM90 300-case getter plus boundary smoke after any change. |
| `DSV4-PREFIX-RESTORE` | `299aaea9` introduced full-prefix DSV4 collection and `50f12ed1` added prefix-resolved Python SDK lookup. `f4d1bcc7` still swept all `_PREFIX_LENGTHS`, but `9a5fe012` incorrectly restricted context collection to prefix 0 with the false claim that the consumer could not resolve prefixes; current work restores the full prefix axis. Rust still assumed historical single-step data, so the current change also threads `prefix` into its table lookup and mirrors Python exact/clamp/interpolation semantics while preserving step-0 compatibility. This is a producer/consumer contract restoration. The separate inner-error policy is tracked below. | Context smoke exercised prefix 0 and 512 through the 0.5.14 request/KV-pool path. The request allocator, positions, SWA tail, and page-rounded pool layout are real, but prefix KV/indexer contents are synthetic and the disabled radix cache omits serving lookup/eviction cost; this proves shape/layout parity, not end-to-end serving parity. Outer getters remain 88 CSA + 88 HCA. The exact source grid changes from 752 prefix-0 rows to 20,392 full-prefix candidates per op, `+19,640`. At first classification, 19,001 CSA and 20,224 HCA were retained only as historical failing references. The strict `3f58b147...` run subsequently completed: CSA is 5/88 done, 83/88 failed, and 19,073/20,392 unique positive keys; HCA is 88/88 with 20,296/20,392 and only 96 source-backed live-pool omissions. Rust DSV4 tests pass 5/5, including multi-anchor interpolation and historical single-anchor fallback. | Regenerate SM100/103/120 prefix/sequence grids rather than copying Hopper capacity limits. Probe prefix allocation, SWA tail reuse, and max-position limits. SM100/103 have bundled FlashMLA paths; SM120 small shapes fall to Triton/PyTorch and its large-shape path remains unresolved. Any Blackwell prefix change must rerun SM90 prefix 0/512 smoke, Python/Rust prefix-resolved queries, and the historical step-0 reverse query. |
| `DSV4-HISTORICAL-BLACKWELL-0510` | `57ba6e3c` (#1034) extended the older SGLang 0.5.10 DSV4 collector to B200/B300/GB platforms. Its runtime-SM-count and TMA-stride corrections remain useful provenance. The same historical line also contained recovery behavior that continued after a poisoned CUDA context, swallowed cleanup failures, and accumulated Hopper-specific FP4 workarounds; those behaviors are intentionally not replayed blindly into this 0.5.14 collector. | This row carries history only. It does not change the current H20 run or make old B200 measurements valid for 0.5.14. Any reused invariant must pass the frozen SM90 smoke/full contract. | The packaged B200/B300/GB evidence under `57ba6e3c` is hardware evidence for SGLang 0.5.10 only. For 0.5.14, SM100/103/120 remain source-derived until rerun. When comparing branches, preserve the proven SM-count/TMA facts while requiring clean worker replacement after fatal CUDA errors and explicit cleanup/error accounting. |
| `DSV4-DECODE-MULTISTREAM-PARITY` | The generic full-module CUDA-graph helper originates at `864b987d` and survives the later `299aaea9` rewrite while SGLang's serving graph remains disabled. `9a5fe012` is the first-known 0.5.14 integration mismatch and has no same-runtime last-good: SGLang's DSV4 layer enables production multi-stream overlap only inside `model_capture_mode`, while AIC's generic `benchmark_with_power` graph does not enter that framework context. The collector therefore measures the sequential branch even though graph capture succeeds. This is known measurement-parity debt, not a reason to weaken graph validation. | Exact-image source sets the non-Blackwell limit to batch 64. Full generation grids and the HCA context grid completed on the sequential collector path; the strict CSA sweep also completed but is an unacceptable failing comparator. These runs do not prove production decode parity. An accepted complete CSA artifact remains pending. A future parity fix must compare batches 64 and 65 in production-capture and collector contexts, then rerun the old sequential case as a reverse check. | Exact-image source sets a 128 limit when `is_blackwell_supported()`, covering the source selector for SM100/103/120. B200 must probe batches 128 and 129 under real `model_capture_mode` before publishing module data; SM103 and SM120 still need their own runnable-leaf proof. Any Blackwell capture change must rerun the SM90 64/65 boundary. |
| `DSV4-TOPK-PHASE-AND-SM120` | `50f12ed1` introduced standalone CSA top-k calibration using only `plan_topk_v2`/`topk_transform_512_v2` and applied the resulting delta to both context and generation consumers. `9a5fe012` narrowed collection support to SM90/100/103 but left that phase mismatch intact; frozen snapshot `cdfca0b9...` therefore remains v2-only and must not produce accepted DSV4 calibration. Exact SGLang 0.5.14 context allocates `c4_sparse_raw_indices` and takes v1, normal SM90/100/103 decode takes v2, and SM120 explicitly disables v2 and takes v1. The current correction, frozen in post-DSA manifest `3f58b147...`, keeps one op/file but writes `v1_flat/v1_top_last` for context and `v2_flat/v2_top_last` for decode, with per-row `topk_transform_v1/v2` provenance. Python and Rust consumers now select only the exact phase/platform variant; a variant-aware file never borrows another variant or legacy rows, while a legacy-only `flat/top_last` file remains backward compatible. Rust had intentionally omitted the correction in `31380331` only because packaged 0.5.10 had no effective calibration; nonzero 0.5.14 rows make native parity required now. | The H20 plan completed 11/11 outer tasks from read-only snapshot `3f58b147...` with zero errors and 4,502 unique rows: 2,051 each of context `v1_flat/v1_top_last` plus 200 each of decode `v2_flat/v2_top_last`. Every shape has its exact score-mode pair; all latency is finite/non-negative, with 2,298 explicit zero-correction rows where `c4_len <= topk_k` and 2,204 positive measurements. CSV SHA is `8a6b0daa...`; the task left no container or GPU process. All 196 cross-phase physical overlaps remain because their kernels differ. Host contracts also pass: phase dispatch/raw-index/v2-plan/source/count tests 21/21 with the noncore suite, Python DSV4 database tests 64/64, Rust DSV4 tests 11/11 including packaged 0.5.10 reverse queries, and Ruff/diff checks. HCA remains uncorrected. | SM100/103 retain context-v1/decode-v2 and require independent hardware rows/source checks. SM120 uses v1 for both phases; Python/Rust consumers are explicit for that future data, but the producer remains gated to `{90,100,103}` until the SM120 whole-module path is validated. When that gate is reviewed, deduplicate same-variant context/decode overlaps rather than blindly emitting duplicate v1 rows. Unknown SMs remain uncorrected. Every Blackwell change must reverse-test both H20 variants, legacy-only loading, missing-variant no-borrow behavior, Python CP's v1 `top_last` fail-loud path, HCA's zero correction, and packaged 0.5.10 native queries. |
| `DSV4-B200-SPARSE-PREFILL-CHUNK` | `50f12ed1` added a device-SMEM-derived cap of roughly 11.6K fresh tokens to avoid the old FlashMLA sched-meta launch failure; both exact-0.5.14 lines (`c8c00f42` and `9a5fe012`) inherited it. Exact SGLang 0.5.14 now switches `PagedIndexerMetadata` to its JIT large/sparse-prefill metadata path when query rows exceed 11,673. The collector applies its older cap first, so a runtime with a 16K prefill chunk can never exercise that framework branch. There is no proven same-path 0.5.14 last-good: `299aaea9` predates the cap, but the old sched-meta failure was real. | H20 derives an 8,192-token chunk, below both thresholds, so the current SM90 plan is unaffected. Record the runtime chunk and require the full H20 row set before closing this side. Do not lower the common plan just to match Hopper. | B200/B300 commonly derive a 16,384-token chunk, so 12,288/16,384-token cells can be silently pruned even though production selects the new branch above 11,673. Replace or scope the cap only after a real B200 probe; compare getter/output deltas and rerun the full SM90 context set. Recompute SM103 from its actual memory/runtime defaults. SM120 has different indexer fallbacks and no source basis for inheriting this old CUDA sched-meta cap. |
| `DSV4-KERNEL-PROVENANCE-AND-MODEL-KEY` | Since `864b987d`, module rows persist the coarse `kernel_source="compressed_flashmla"`; it cannot distinguish normal SM90/100/103 FlashMLA, the >11,673 large/sparse-prefill path, or SM120 Triton/PyTorch leaves. Standalone CSA/HCA microbench pools also use a generic page size 64, while production layouts are main/SWA 256, CSA extra 64, and HCA extra 2. Separately, the full-module schema created by `299aaea9` persists model metadata but omits model identity, hidden size, and Q-LoRA width from the consumer lookup key, so Flash and Pro rows can collide. There is no safe multi-model last-good schema. Current work fails closed to one canonical full/raw model rather than appending both. | The frozen H20 plan uses only `sgl-project/DeepSeek-V4-Flash-FP8` for DSV4 module/calibration data; Flash and Pro remain separately covered where the operation key distinguishes execution, such as MoE. All four collected strict-baseline files have unique physical and consumer keys with no outside rows. Final acceptable CSA completeness remains pending. Treat the coarse source as a family label, not proof of a leaf kernel. | Do not append Flash and Pro module rows into one B200/SM103/SM120 file. Either keep the canonical profile or add reviewed producer/loader/query dimensions with Python and Rust migrations. Probe and record the actual leaf and page layout per platform; never reinterpret old `compressed_flashmla` rows silently. Any schema change must still load historical single-model SM90/H200 data without overwrite. |
| `GLM-SKIP-INDEXER` | `f4d1bcc7` introduced GLM-5.2 and the forward-hook/indexer-output bypass. `9a5fe012` inherited it unchanged and is the first-known-bad 0.5.14 runtime because production carries indices through the attention module's `next_skip_topk` return; no same-runtime last-good exists. The current return-value path is a new-runtime correction. The local cached config also normalizes legacy `deepseek_sparse_attention` to Transformers-v5 `compressed_sparse_attention` without changing architecture or `index_topk`. | Collector reuses producer top-k; context/generation centralized smoke passed 8/8 + 8/8. Dense `MHA_ONE_SHOT` prefill legitimately has no indexer. Full strict targets are 17,616 context and 1,224 generation skip rows. | Probe the actual SM100/103 selector and return-value path for short/long context and decode; mark SM120 explicitly unsupported or validated rather than inheriting the condition. Rerun both SM90 full/skip shared-file counts after any change. |
| `GLM52-DSA-1M-CEILING` | Batch 32 entered the common module plan at `1d12d321`; the `total_kv_tokens <= 1 << 25` guard came from `299aaea9`; and `2ae0a2ae` added the generic per-model exact-max-position sampler. `f4d1bcc7` is the first commit that combines those pieces with GLM-5.2 and its 1,048,576 ceiling, so its parent `ad0e3e5e` has no executable GLM-5.2 1M family rather than a same-path passing case. `9a5fe012` is the first exact-0.5.14 exposure and only inherits the shapes. Neither B200 auto-healing `f027123f` nor L40S auto-healing `29fae326` touches this limit, ceiling, or model config, so they are not introducers. Every full-run failure was reported when `batch_size * (prefix + isl) == 2^25`, making equality a useful observation boundary, but the original CUDA error was reported asynchronously. | The full H20 sweep persisted 48,531/48,560 rows. BF16 KV outer tasks failed with 64/32 local heads (TP1/2), while 16/8 (TP4/8) completed; FP8 failed with 64/32/16 (TP1/2/4), while 8 completed. The five failed groups miss 29 exact-ceiling rows. Fresh `CUDA_LAUNCH_BLOCKING=1` single-shape reruns all pass: full BF16 TP1 exact and adjacent-below, full BF16 TP4 exact, and full FP8 TP8 exact each persist 1/1. Complete ordered blocking reruns then reproduce the original per-task cutoffs exactly: BF16 TP1 persists 204/210 and fails at exact `isl=8`, BF16 TP2 persists 208/210 and fails at exact `isl=128`, while FP8 TP1 and TP4 persist 203/210 and fail at the first exact point `isl=4`. The blocking stack identifies SGLang `dsa_indexer._get_topk_ragged -> get_index_k_scale_buffer -> index_buf_accessor._get_k_and_s_triton_kernel`, not the earlier asynchronous `repeat_interleave` site. BF16 TP1 even persists the first exact point before the second exact point fails in the same process. A fresh BF16 TP1 subprocess containing only those first two exact points passes 2/2, so consecutive exact shapes alone are insufficient; some part of the preceding 203-shape history is required. Equality is therefore not a deterministic standalone shape limit; the evidence is a configuration-dependent within-process state/sequence defect. No `< 2^25`, KV-backend, TP, or shape skip is justified. | Do not copy a Hopper skip to Blackwell. On SM100/103, rerun both a clean exact single case and the complete ordered batch-32 outer sweep, recording the actual DSA/indexer backend, every persisted exact point, and the first failing point. SM120 does not emit this DeepGEMM DSA plan today. Any eventual fix must preserve every isolated equality success plus the original TP4/BF16 and TP8 full-sweep controls; neighboring 1,000,000-prefix points alone are insufficient to detect sweep-state pollution. Re-run both the two-point consecutive-exact diagnostic and a complete ordered sweep after any Hopper or Blackwell state-reset change. |
| `GLM52-SKIP-RETURN-CEILING` | `f4d1bcc7` introduced GLM-5.2 skip-indexer collection. Its original collector implementation captured an indexer forward-hook value; the current uncommitted 0.5.14 correction instead uses SGLang's production `next_skip_topk` second-return contract for one producer warmup, then passes that tensor as `prev_topk_indices` while timing the reuse layer. Frozen snapshot `cdfca0b9...` contains this correction. Exact-image source SHA `49e384ce` proves the return/reuse contract, but the shallow image history does not expose its upstream introducing commit. The skip path is not equivalent to removing the producer entirely: every shape must first execute one real producer forward. | The completed H20 skip-context plan has 88 outer tasks and eight failed batch-32 IDs covering BF16/FP8 × TP1/2/4/8. Seven IDs each persisted 2,195/2,202 rows and reported an asynchronous illegal access at the exact ceiling; FP8 TP1 instead hit one `EADDRINUSE` before its batch-32 task and is missing all 210 rows. Fresh blocking single-shape reruns contradict a fixed boundary defect: skip BF16 TP4 exact/below and skip FP8 TP8 exact/below all persist 1/1, while their full exact controls also pass. Complete ordered blocking reruns make the split explicit: skip BF16 TP4, BF16 TP8, and FP8 TP8 each persist 203/210 and fail at the first exact point inside the same SGLang fused accessor; skip FP8 TP1 completes 210/210. A separate stock, non-blocking FP8 TP1 recovery also completes 210/210 and exits zero, proving `EADDRINUSE` was transient without reusing blocking latency. The immutable frozen file remains 17,357/17,616 skip rows; the separate reconciled artifact has 17,567/17,616 skip rows and exactly 49 accepted fused-accessor gaps. Combined full+skip is 66,098/66,176 with 78 disclosed gaps, zero key collision, and exact Python loader leaf counts. Generation skip remains 8/8 and 1,224/1,224. No equality, backend-wide, or skip-indexer-wide exclusion is justified. | SM100/103 must probe producer-return plus reuse separately from ordinary full attention using both isolated equality points and complete ordered sweeps. Record the actual TRT-LLM/FlashMLA selector and whether returned indices remain valid across repeated shape transitions. SM120 does not emit this plan today. Any fix must preserve H20 isolated full/skip equality and below-boundary successes, the FP8 TP1 210/210 ordered control, generation 8/8, and the production `next_skip_topk` contract; do not restore the old hook or add an equality skip merely to hide sweep-state pollution. Re-run the same full/skip ordered pair on Blackwell so a platform-specific reset is not generalized silently. |
| `DSA-FUSED-KS-4G-OFFSET` | Upstream SGLang `043f1317` (#13812) introduced the fused Triton K/S accessor; `006bd44c` (#19319) rewrote it for batched long sequences. Exact 0.5.14 source SHA `49e384ce` retains pointer offsets such as `page_index * buf_numel_per_page` and `output_token * index_head_dim` without an explicit 64-bit cast. AIC `299aaea9` later admitted aggregate KV equality at `2^25`, `2ae0a2ae` added exact model ceilings, and `f4d1bcc7` supplied GLM-5.2's 1,048,576 limit; `9a5fe012` merely exposes their combination. Neither B200 `f027123f` nor L40S `29fae326` introduced this arithmetic. At batch 32 the exact family produces `2^25` K tokens and a `2^32`-byte K output; the packed K+scale source span is `2^25 * 132 = 4,429,185,024` bytes. | The 4 GiB crossing matches every failing point and the blocking stack; all 203 preceding points remain below it. Fresh exact processes and a fresh two-exact sequence pass, while complete ordered processes fail configuration-dependently. An instrumented stock BF16 TP1 ordered sweep (`dsa_state_diag/full_bf16_tp1`, diagnostic SHA `0904d7dc...`) reproduces 204/210 and the exact `isl=8` crash. Both exact points have valid metadata: page table `(32,16384)`, 524,288 used entries, min/max page `1/524288`, 550,503 buffer pages, zero invalid indices, 35,232,128 allocator tokens available, and about 32.6 GiB free at the kernel. The first exact point passes and the second fails with the same bounds, excluding pool capacity and out-of-range page metadata. A task-local safe-offset comparator (`fc4e491f...`) logs `page_indices=int32`, `seq_lens=int64`, and completes 210/210; output SHA is `8e75ab8e...`. It combines explicit int64 source/output offsets, defined masked-load values, stricter valid-page masks, and a new Triton cache identity. Therefore it proves the stock fused accessor is the failing component, but does not isolate int32 overflow from masked-lane or codegen/cache behavior and does not prove K/S value correctness. No framework patch or patched latency is added to AIC. | Run the identical stock and safe-offset ordered sweeps on SM100/103 before transferring the classification; SM120 has no emitted plan today. Report/upstream the fused-accessor defect with the caveats above, and keep the 0.5.14 release-image gaps disclosed rather than shipping an AIC kernel fork, process-per-shape workaround, reordering, or `<2^25` skip. Any upstream-quality fix needs mask/int64/cache ablations plus small-shape value comparison against the reference accessor. Any future SGLang image update must rerun the stock H20 sweep, isolated exact controls, and packaged artifact validation. |
| `DSA-ATTN-TP-CONTEXT-LIFETIME` | Since the original SGLang DSA collector commit `57378df4`, AIC calls `get_attn_tp_context().set_attn_inputs(...)` but does not call `clear_attn_inputs()` at the end of each shape. Exact SGLang 0.5.14 production `deepseek_v2.py` does clear it after the model forward. `299aaea9`, `c8c00f42`, and the current frozen snapshot retain the collector mismatch; it is not a Blackwell change. | This remains a secondary parity/lifetime concern, not the accepted 4 GiB failure: valid page/pool diagnostics and the task-local safe-offset accessor isolate the stock fused kernel without changing this context lifetime. No cleanup change is mixed into the DSA acceptance or current collector snapshot. A future correction must measure allocator/live-memory state and rerun the complete ordered sweep plus normal short-prefix controls rather than being added as an unproven reset. | Production clears the context on every platform, so any eventual collector correction should be platform-neutral. Validate SM90 first, then repeat memory-state and ordered-sweep checks on SM100/103; do not use the cleanup to conceal a still-failing fused kernel or infer SM120 support. |
| `DSA-SUBBACKEND-SELECTOR` | `9a5fe012` introduced the FP8-KV rule `SM90 -> flashmla_kv`, `SM>=100 -> trtllm`; it mirrors SGLang 0.5.14's major-based selector but overstates runnable hardware. The SM120 whole-module skip is a separate inherited change from `37826f10`, originally backed by an SGLang 0.5.10 RTX PRO 6000 run. Full and skip-indexer share the same attention backend; only indexer reuse differs. Persisted `flashmla_kv/trtllm` is a configured-backend bucket, not an exact leaf-kernel label for every shape. | On SM90, explicit FP8 KV selects `flashmla_kv`; short eligible context may still execute dense FA3, while long context/decode use the sparse path. The current full run must contain no `trtllm` bucket. Producer GEMM identities are DSV3.2=`fp8_block`, GLM-5.2=`bfloat16`, GLM-5-FP8=`fp8_block`; KV dtype is an independent axis. BF16 KV is left on framework defaults. | Exact-image source proves SM100 TRTLLM-GEN is runnable. SM103 gets the same selector name, but the bundled capability test rejects SM103 TRTLLM-GEN; bundled FlashMLA has an SM103 binary and is only a source-derived alternate. SM120 is forced toward TRTLLM-GEN, has no bundled FlashMLA target, and remains skipped/unvalidated for 0.5.14. Preserve these as three distinct states. `next_skip_topk` is backend-independent after absorbed attention; `MHA_ONE_SHOT` produces no top-k and must remain the explicit exception. |
| `DSV32-SM100-REDUCED-HEAD-DECODE` | B200 auto-heal `f027123f` added a Blackwell-only DeepSeek-V3.2 generation exclusion for `head_num <= 32` and `kv_len >= 256` after an asynchronous illegal access/SIGABRT. `2ae0a2ae`, `f4d1bcc7`, and divergent `c8c00f42` retained the whole-family skip; `9a5fe012` removed it. This is an explicit Blackwell add-then-remove cycle, not a Hopper rule and not evidence that the 0.5.14 path is fixed. | SM90 never selected this SM100 predicate. The current H20 DSA evidence therefore cannot validate or invalidate either side of the cycle. No skip is copied into the SM90 plan. | On B200, run the four boundary families `(heads=32/64, kv=128/256)` with the exact 0.5.14 decode selector in fresh workers, then continue the full outer task if the boundary passes. Record the first fatal shape and nearest success before considering any replacement predicate. SM103 and SM120 require independent selectors and must not inherit the SM100 skip. |
| `DSA-PREFILL-GRAPH-BOUNDARY` | The benchmark boundary changed repeatedly: `299aaea9` enabled GLM DSA piecewise graph by default; `f027123f` made it explicit-env only; `f4d1bcc7` re-enabled it for the TRT-LLM prefill backend while keeping SM90 `flashmla_kv` eager; and both `9a5fe012` and divergent `c8c00f42` force the module path eager. These are measurement-contract transitions, not harmless compatibility edits. | The accepted H20 full/skip data measures the eager `flashmla_kv` module path. It does not validate the historical piecewise-graph numbers, and no graph result is merged into the current artifact. | B200 must identify the actual 0.5.14 prefill backend first, then compare eager and any proposed graph boundary on the same short/long shapes. Do not revive the 0.5.10 TRT-LLM graph choice from commit history alone. Any boundary change requires a new namespace/snapshot and a reverse H20 eager control. |
| `GLM-SPARSE-EXTREME` | `2ae0a2ae` first introduced standalone GLM sparse MQA/top-k with four coupled defects: monolithic logits/score allocation, the incorrect `isl > 1 => ragged fused` context assumption, present all-zero decode `row_starts` although production omits it, and pad4 rather than page/block-rounded pad64 decode score stride. Exact 0.5.14 `topk.cu` dispatches on optional presence, so the decode call selected the prefill leaf. `299aaea9` already knew the full-module MQA path could chunk, but predates this file. B200 auto-heal `f027123f` has zero direct `glm5_dsa_sparse_modules.py` diff; its later-consumed shared MLA inputs alter none of these contracts. The defects are inherited/not introduced by Blackwell, not a zero-diff claim for every plan input. `f4d1bcc7` exposes them to GLM-5.2's 1M ceiling; `9a5fe012` inherits them. Snapshot `3f58b147...` fixes only MQA. Ragged snapshot `89372408...` fixes geometry/allocation but preserves the three selector/stride mismatches. Mutable `c5cba92d...` fixed context only; `8f05cbfd...` also omitted decode `row_starts`; `6da3683b...` also corrected decode stride. Each was superseded in review without GPU execution. Current pre-freeze `1194e6b6...` additionally releases each materialized chunk page table before allocating the next score/table pair; that is peak-memory hardening, not a reproduced correctness failure. DSV3.2 wrapper `383e2a59...` changes prose only and remains unregistered. | Under immutable `3f58b147...`, MQA is 11/11 and 3,332/3,332, while strict top-k is the rejected 3/11, 6,520/6,664 monolithic comparator with 72 physical OOMs and 144 paired gaps. Snapshot `89372408...` is row-complete at 11/11 and 6,664/6,664, with CSV `e327b330...`, checkpoint `da98312d...`, and key hash `e714eac7...`; structural validators `36662f4f...` and `1bd717b1...` PASS. Exact source/H20 logs invalidate all 6,376 context rows because the retained profiles are PAGED. Of 288 decode rows, 214 are select-all zeros and return before a kernel launch; all 74 nonzero rows selected the wrong leaf, and 68 of those rows (34 shapes) also had wrong stride. The other six nonzero rows are three injected ceilings that happen to be 64-aligned. Exact `topk.cu` SHA256 `f899cb9d...`, lines 488--505, proves the optional-presence dispatch; exact `dsa_indexer.py` lines 601--620 and 704--724 prove page-size-64 rounding. The context-only candidate passed an incomplete 36/36 suite; the decode-omission candidate passed its then-current 37/37 suite (`e41b8657...`) before stride review superseded it. Current source/test hashes `1194e6b6...`/`89026f7d...` pass final host review 50/50 plus Ruff/format/diff checks. At that pre-GPU review checkpoint these were host contracts only and no corrected GPU evidence existed; the subsequent selector, value, and stress results are appended below. | The required SM90 sequence was context/decode value and leaf oracles, persisted PAGED source, expanded-page-table peak memory, and then a fresh 6,664-row run. Frozen `1194e6b6...` now passes the selector/value/leaf gates and all eight stress shapes, including batch-one and logical 16-TiB controls; the full recollection and a validator that rejects the row-complete mixed wrong-path artifact remain pending. SM100 FP8 defaults to `trtllm`/PAGED and needs B200 validation of the same optional/stride/lifetime contracts. SM103 has separate capability debt; SM120 is source-selected PAGED but plan-excluded. Reverse-test the H20 extreme and neighbors, equal-40-GiB pair, stress tuple `(32,4096,1044480)`, a non-64-aligned decode plus the three aligned ceilings, MQA 3,332, and key hash. The stress tuple is 512 GiB under rejected compact `[Q,S]` and 16 TiB under bounded concatenated `[Q,bs*S]`; do not add skips or copy latency to Blackwell. |
| `GLM-MQA-RAGGED-PAGED-BOUNDARY` | Current uncommitted chunking corrects `2ae0a2ae`'s standalone collector for SGLang 0.5.14's canonical ragged prefill path; it is not a change introduced by `f4d1bcc7` or `9a5fe012`. The 3,188 retained context shapes use the production ragged MQA ABI and a query-row memory bound, but the collector rereads free memory per shape rather than mirroring serving's cached/static budget. The 144 retained `isl=1` rows still call non-paged MQA, while production decode/idle uses `_get_topk_paged`; they are frozen artifact-completeness rows, not production-decode evidence. | The strict MQA full run completed 11/11 with all 3,188 context plus 144 legacy decode-shaped rows, for 3,332/3,332 unique keys and zero error. Keep those 144 rows for this frozen release plan, but do not use them to claim paged decode coverage. The separate top-k monolithic-score failure is recorded in `GLM-SPARSE-EXTREME`; it does not invalidate the MQA file or justify copying a gap into it. | Revalidate the live-memory budget independently on SM100/103. SM120 first needs a supported alternate backend. If paged decode is collected later, make it a distinct reviewed contract instead of silently changing existing keys, and reverse-test all SM90 ragged keys plus legacy non-paged rows. Hopper's former top-k omission and MQA budget must not be copied to Blackwell. |
| `DSV32-SPARSE-SHARED-ORPHAN` | `cd084d20` introduced a DeepSeek-V3.2 wrapper that directly delegates MQA/top-k to the GLM worker. B200 auto-heal `f027123f` predates that wrapper and has zero direct wrapper/registry diff; it did change the common MLA inputs later inherited by the worker, so this is not a claim of zero diff across every shared dependency. Because the wrapper remains unregistered in the SGLang registry/PerfFile/YAML, those inputs emit no DSV3.2 sparse task; the orphan is inherited/not introduced by Blackwell. It has no local artifact and defaults to an `-Exp` model ID without a matching local config. Do not fix or register this adjacent debt as part of the GLM correction. | The active SM90 plan has zero DSV3.2 sparse tasks, so GLM full green cannot claim DSV3.2 support. A shared GLM helper change must keep a wrapper/decode-selector contract and, if directly probed, use the actual `deepseek-ai/DeepSeek-V3.2` model ID without publishing rows. No DSV3.2 full rerun is required while the path remains unregistered. | Future enablement must add an explicit SM/version/model gate and collect fresh data rather than reuse GLM rows. Source-derived counts are MQA 2,931 and top-k 5,862 rows over 11 batch tasks; SM100/103 require independent runs, while SM120 cannot inherit this DeepGEMM path. Any future registration must first resolve the model ID and run small, long multi-chunk, and decode-selector controls on SM90. |
| `INNER-SWEEP-FAILURE-GATE` | Clean `9a5fe012` was inconsistent: DSV4/GDN/GLM failed on any inner error while mHC could silently return partial or empty data. The first uncommitted correction applied `failed * 3 >= attempted` to DSV4, GDN, GLM sparse, and mHC; frozen snapshot `cdfca0b9...` contains that policy. Review showed that this incorrectly turned the user's investigation heuristics into automatic acceptance. The post-freeze working tree restores fail-closed behavior for every unexpected inner error, keeps source-backed capacity skips separate, and removes the single-use GDN rate helper. This is a checkpoint-status change, not a measurement-row change. | The frozen GDN checkpoint reports all 16 outer tasks done because of the old one-third gate, but its artifact still has exactly 1,846/1,848 rows: the two batch-1024 packed-decode failures in `GDN-DECODE-BS1024` remain manually disclosed and accepted, not automatically green. A rerun with the current tree must mark those two outer tasks failed unless they become explicit, narrowly reviewed expected cases. The strict DSV4 baseline is complete (CSA failing comparator 5/88 and 19,073/20,392; HCA context 88/88 and 20,296/20,392; both generation files 1,600/1,600), and mHC completed 4/4 with 140/140. At snapshot `3f58b147...`, GLM MQA and DSA-attention are each 11/11 and 3,332/3,332 while top-k remains the rejected 3/11, 6,520/6,664 monolithic comparator. Snapshot `89372408...` is structurally row-complete but has 6,376 wrong-transform context rows, 74 wrong-leaf nonzero decode rows, and within those 68 wrong-stride rows. Mutable `c5cba92d...`, `8f05cbfd...`, and `6da3683b...` are superseded review checkpoints. At the pre-GPU review checkpoint, `1194e6b6...` passed host review but remained GPU-unvalidated; its later selector/value/stress results now pass and are appended below. The remaining GLM work is the fresh 6,664-row top-k run, plus the separate accepted-complete CSA decision; MoE-last is tracked independently. | Do not add a platform-specific percentage gate. On SM100/103/120, classify each missing point with its nearest success and hardware path; only an explicit source-backed expected/capacity predicate may be excluded from failure accounting. Any such predicate must be replayed against the exact SM90 failure and successful boundary controls before updating this row. |
| `MHC-TEARDOWN-AND-PARTIAL` | Clean `9a5fe012` catches per-shape OOM/runtime errors, returns partial data, and only deletes the runner/empties the allocator cache; it does not destroy SGLang model-parallel groups or the process group. The first uncommitted correction added teardown but swallowed every `destroy_model_parallel()` exception and used the one-third gate; frozen snapshot `cdfca0b9...` contains that state. Exact-image SGLang 0.5.14 `destroy_model_parallel()` already checks each optional group before destroying it. Post-DSA snapshot `3f58b147...` instead lets teardown exceptions fail the worker, destroys the process group when initialized, and rejects every unexpected partial sweep. | The H20 full retained run from snapshot `3f58b147...` completed 4/4 outer tasks with zero errors and 140/140 unique positive-latency rows across `pre` and `post`; CSV SHA is `0a2070cb...`. Logs contain no teardown warning, CUDA error, OOM, or traceback, and all four worker allocations disappeared after the task container exited. Host contracts separately prove both one isolated kernel failure among four points and an injected teardown failure propagate; the full noncore file passes 21/21. A worker failure remains preferable to retaining hidden groups/memory for the next task. | SM100/103/120 can initialize a different set of TP/EP/attention groups, so validate successful cleanup and a forced teardown failure on each runnable platform rather than adding `except: pass`. Any Blackwell cleanup change must preserve the H20 isolated-failure and teardown-propagation controls and leave process reclamation to the failed task worker. |
| `MHC-PRENORM-ENV` | B200 auto-heal `f027123f` set `SGLANG_OPT_DEEPGEMM_HC_PRENORM=0` inside the mHC module. `50f12ed1` then set the variable to `1` in central `collect_sglang()` before registry import, so that earlier module-level `setdefault(0)` no longer won; its B300 report changed from roughly 53% slow to mean 2.5%/max 7.9%. Divergent `c8c00f42` and current base `9a5fe012` remove the module default and retain central `setdefault(1)`. External environment can still override every state. | Source entry expects `1`, but the completed H20 log printed only `tilelang_pre=1` and did not print the actual prenorm environment value. The 140/140 artifact is hardware evidence for the process state used, not direct proof that the intended env value won. | B200/SM103/SM120 runs must print the resolved env value before import plus the selected kernel. Any change must rerun H20 pre/post mHC controls; do not infer the value from `setdefault` source or merge rows from different resolved values. |
| `GDN-DECODE-BS1024` | `9a5fe012` replaced the old simulation with SGLang 0.5.14 packed QKV and `fused_recurrent_gated_delta_rule_packed_decode`; the pre-`9a5fe012` non-packed collector is not a valid last-good. Exact image source SHA `49e384ce` confirms collector ABI/dtypes/state layout match production. The kernel launch `grid=(NV, B*HV)` came from upstream SGLang commit `f4346f0a` (PR #20627), not this branch's Blackwell work. For the two large models, `B=1024, HV=64, V=128` gives `NV=4` and `gridY=65,536`, one above CUDA's 65,535 limit. AIC intentionally stores an unsharded/TP1-equivalent GDN table; production TP>=2 can have local `HV<=32` and need not hit this point. | All 16 outer tasks completed and 1,846/1,848 rows were persisted. Only Qwen3.5-122B-A10B and 397B-A17B packed decode at `batch_size=1024` fail, each 1/11 decode points, with `Triton Error [CUDA]: invalid argument` on clean GPUs. Keys are unique and positive; the two failures are disclosed rather than called error-free. | When Triton packed decode is selected, SM90/100/103/120 share this grid-Y boundary. SM100/103 may instead auto-select FlashInfer only under their specific BF16-state conditions; SM120 does not inherit that auto-selector. Reattempt and record the actual selector on every platform, but do not relabel this as OOM, copy the omission blindly, or change collector layout to mimic a sharded deployment. Any kernel/layout edit must rerun both H20 shapes and a neighboring success. |
| `MOE-ONE-RUNTIME-TRUTH` | The current YAML/model-case routing makes four quantized families enter SGLang 0.5.14 `FusedMoE` directly, leaving the old low-level Marlin, TRT-LLM BF16×FP4, CuteDSL NVFP4, and MXFP4 branches unreachable. The old MXFP4 branch/parallel patch came from B200 auto-healing `f027123f`; the old CuteDSL NVFP4 path came from `97faa88f`; the Marlin branch predates this release; and `9a5fe012` introduced the now-unused `_framework_routing` / `_resolve_framework_moe_backend` tables. Keeping those beside the new YAML resolver created conflicting Hopper/Blackwell truths: 12 of 35 models disagreed, including Gemma GELU, Llama4 routing, and DSV4 correction bias. Post-DSA snapshot `3f58b147...` removes only those unreachable paths/tables/tests and inlines exact-0.5.14 `MoeRunnerConfig` construction with its five supported fields. The active graph is YAML `get_sglang_moe_backend` → `run_moe_torch` → high-level `FusedMoE` for quantized/non-default routing, while default BF16/FP8 Triton stays on the raw fused kernel. | This is a net reduction of 652 collector lines and 113 obsolete test lines, not a new compatibility layer. MoE focused tests pass 34/34, model-case tests 65/65, the broader targeted suite passes 388/388, and Ruff/format/diff checks pass. Existing targeted H20 Kimi INT4/DSV3 FP8/Nemotron NVFP4/DSV4-Pro W4A16/Gemma BF16/GLM NVFP4/GPT-OSS MXFP4 smokes predate the cleanup, so rerun representative raw Triton and all four high-level quant modes from snapshot `3f58b147...` before full MoE. | On SM100/103, validate `flashinfer_trtllm` and `flashinfer_mxfp4` through the same high-level path; do not resurrect `f027123f`'s private MXFP4 patch. SM120 remains a separate Marlin/CUTLASS audit. Kimi INT4 is intentionally absent from the SM100 plan even though a direct backend getter would fall back to Triton; preserve the zero-population plan test. Every Blackwell backend edit must rerun the H20 YAML mapping, physical-key collision guard, raw BF16/FP8, Kimi Marlin, and DSV4-Pro W4A16 controls. |
| `MOE-BACKEND-ARTIFACT` | The `sglang_moe_*` YAML fields and `get_sglang_moe_backend` first appear on divergent commit `c8c00f42`, not clean `9a5fe012`. Separately, B200 auto-healing `f027123f` added a private MXFP4 patch, while `9a5fe012` added much of a second 0.5.14 framework-quantized runtime truth. The YAML/backend matrix and `collect_moe.py` cleanup frozen in snapshot `3f58b147...` reconcile those lines around the exact SGLang 0.5.14 high-level `FusedMoE` path. They preserve checkpoint-native quantization and reject physical-key collisions with different execution signatures. This is not one linear regression and must be updated per artifact/backend; the exact tracked patch identity is `d8cc98aa...`. | Eight targeted paths pass: Kimi INT4, DSV3 FP8, Nemotron NVFP4, DSV4-Pro W4A16, Gemma4 BF16, GLM-5.2 NVFP4, GPT-OSS MXFP4, plus a shuffled sample. Current full target is 114,105 unique keys: BF16 45,684; FP8-block 38,799; NVFP4 20,493; W4A16 6,051; INT4 3,078. Full run is pending. Kimi INT4 is SM90 Marlin with group size 32. | Source-derived only here: Kimi FP8 uses `flashinfer_trtllm` on SM100/103; DSV4 W4A8 uses `flashinfer_mxfp4` on SM100/103; SM120 exceptions must be regenerated and probed separately. Never infer a backend from the old CSV label or resurrect the private MXFP4 patch. After any Blackwell backend change, rerun SM90 getter quant counts, zero-population Kimi INT4 on SM100, Kimi Marlin, DSV4-Pro W4A16, raw Triton plus all four high-level quant paths, and the physical-key collision guard. |
| `SM100-ALLTOALL-TEST-DEBT` | First-bad `86911fab` (#662) added `num_gpus_per_node >= 72` to the exact-SM100 TRT-LLM AllToAll gate; its parent `a07261e61a38` is last-good for the seven 8-GPU/node mock tests. Exact history is 27 passing before versus seven failures immediately after; clean `9a5fe012` still has them. | Unrelated to SGLang SM90 collection; no code change in this effort. | Decide explicitly whether 8-GPU SM100 uses NCCL fallback or TRT-LLM AllToAll. SM103 and SM120 are separate audits because the existing gate/debt is `_sm_version == 100`; do not extend the conclusion by analogy or use inherited failures to justify a collector workaround. |
| `DSV4-OLD-GATE-HIDDEN-GAPS` | Frozen snapshot `cdfca0b9...` used the first uncommitted one-third inner-failure gate, so an outer task with a few failed shapes was written done. Post-DSA snapshot `3f58b147...` deliberately fails every unexpected inner error. The old June 28 context files must therefore be re-read at key level rather than treated as a clean baseline or copied into Blackwell expectations. | Current static enumeration has 20,392 CSA and 20,392 HCA context keys. Each old file has 96 logged `KVPoolCapacity` omissions, but old CSA additionally misses 1,295 runtime-error cells (860 OOM, 363 `InternalError`, 72 `AcceleratorError`) across 83/88 outer tasks; old HCA additionally misses 72 `AcceleratorError` cells across 72/88 tasks. The three old failed CSA batch-1024 outer tasks cover only 75 cells, leaving 1,220 extra unapproved CSA gaps hidden behind the gate. Therefore old headline counts 19,001 and 20,224 are evidence of the former behavior, not acceptance targets. The strict `3f58b147...` rerun subsequently completed: CSA is 5/88 done, 83/88 failed, and 19,073/20,392; HCA is 88/88 and 20,296/20,392 with only the 96 source-backed omissions. These later results append to rather than erase the old hidden state. | Never make B200/SM103/SM120 reproduce Hopper's old row counts or checkpoint status. Regenerate the platform's 20,392-key-equivalent plan, separate static/source-backed skips from runtime errors, and compare exact missing keys and selected leaves. Any capacity predicate or pool-budget change must reverse-test the same H20 failures and nearby successes under strict accounting; do not restore a percentage gate to make either platform green. |
| `DSV4-DERIVED-POOL-WORKSPACE` | Chronology is `864b987d` (pre-Blackwell full-module collector with fixed `mem_fraction_static=0.7` and `GLOBAL_MAX_PAIR * 16`, i.e. 16M tokens) → `57ba6e3c` (Blackwell extension with zero pool/prefix-setting delta; its old cleanup/error handling could preserve partial rows) → `299aaea9` (renamed/reworked full-module collector, inherited the fixed cap, and consumed the 28-value prefix list) → `f027123f` (exact zero diff in `collect_dsv4_attn.py` and the DeepSeek-V4 model input; its B200 SGLang-0.5.10 parquet has only 765 `step=0` rows and therefore no nonzero-prefix pool evidence) → `50f12ed1` (per-worker swept-KV maximum plus 5%, still pinned to SGLang 0.5.10) → independent `max_total_tokens=None` introductions on divergent `c8c00f42` and `9a5fe012`; `9a5fe012` separately regressed context to prefix zero → current work restores all 28 YAML prefix values without changing that YAML list, yielding immutable failing `None` snapshot `3f58b147...`. This is inherited/not introduced by Blackwell. The earlier bounded implementations predate pinned 0.5.14 or lack comparable full-prefix evidence, so a workspace-aware cap is new diagnostic behavior, not a proven last-good reapplication. Snapshot `3f58b147...` and current worktree have identical collector SHA256 `9d9776c4...`. Source re-audit also corrects an earlier shorthand: exact SGLang 0.5.14 uses `ReqToTokenPool`, `SWATokenToKVPoolAllocator`, `DeepSeekV4TokenToKVPool`, and `DSV4PoolConfigurator`, not `HybridReqToTokenPool`. | In the completed strict H20 run, TP1/BF16 CSA bs8, 16, and 32 reproduce the old June 28 failures 9/9. The exact `(isl,prefix)` triples are bs8 `(1024,524288),(1024,1000000),(512,1000000)`; bs16 `(512,524288),(512,1000000),(256,1000000)`; bs32 `(256,524288),(256,1000000),(128,1000000)`. SGLang derives `mem_fraction_static=0.907`, leaves `max_total_tokens=None`, allocates roughly 134.3–134.8 GiB across its full/SWA/C4/C128/state pools, and then the real C4 logits path requests 4.01, 7.64, or 3.82 GiB with only 0.67–3.81 GiB free. Matching half-workspace neighbors and the same 8,192 fresh tokens at prefix 262,144 pass. This is a collector static-pool/workspace budget mismatch, not an accepted H20 capacity limit. No bounded-pool candidate has been applied or measured yet; any implementation requires a new frozen snapshot and separate namespace. | Do not convert these points into an SM90 or Blackwell skip. First rerun the exact failures and neighboring successes with one explicit, workspace-aware bound that still lets SGLang's DSV4 configurator derive the full/SWA/C4/C128/state pools on H20; only a value-correct, complete sweep can justify changing the collector. Then repeat pool capacity, free-workspace, selected DSV4 leaf, and exact-key checks on SM100/103. SM120 has a different whole-module path and must not inherit either the pool formula or the skip set. Any adjustment requires a new source snapshot and a reverse run of the original `None` state so the add/revert history remains visible. |

### Dense-attention collector-only reversal (2026-07-04)

The `c8c00f42` proposal that promoted dense-attention `kernel_source`,
`v_head_dim`, and `attention_chunk_size` into Python/Rust lookup keys and
EngineSpec v2 is rejected for this collector-only effort. The release-owner
decision is:

- `kernel_source` remains producer provenance and is not consumed by queries;
- V-head, runtime-window, chunk, sink, and scaling values configure the exact
  SGLang 0.5.14 invocation but do not become database dimensions; and
- SDK, Rust, EngineSpec, and untouched framework collectors remain unchanged.

Population deduplicates on the historical attention key plus source and fails
closed if that pair requires different effective runtime semantics. For global
Llama-4 attention, chunk 8192 is retained in the real model invocation but is
not an effective leaf-kernel distinction when `window_size == 0`; this removes
the 357 duplicate H20 context rows produced by the superseded snapshot. The
old 51,258-row context artifact has only 50,901 unique historical-key/source
rows and is execution evidence, not an accepted artifact for this design.
Generation remains 40,468 rows. A fresh collector-only snapshot and smoke are
required before either count is attached to a product commit.

SM100/103 currently contain cross-source overlaps under the historical query
key. Raw source provenance may be collected, but those Blackwell artifacts are
not consumer-ready until a separate, explicitly authorized consumer contract
is designed. This reversal does not alter the historical chronology recorded
in `ATTN-VARIANT-KEY-0514` or `ENGINE-SPEC-V2-ATTENTION`; it supersedes their
present-tense product status for this branch.

### GDN consumer reversal (2026-07-04)

The original SGLang 0.5.14 upgrade changed the Python and Rust Qwen3.5 model
and GDN SOL paths to consume
`fused_recurrent_gated_delta_rule_packed_decode`. That query-side change is
rejected for this collector-only effort. The collector still records the
packed SGLang 0.5.14 kernel source and its exact invocation, but Python/Rust
models and lookup behavior remain byte-for-byte equal to `main`. Consuming the
new source is a separate reviewed SDK change with its own packaged-data tests.

This reversal has no collector plan or execution-path delta, so it does not
invalidate the existing H20 GDN measurements. Those measurements remain
collector evidence only: 1,846/1,848 rows, with the two disclosed batch-1024
grid-Y failures. They do not establish that the current SDK consumes the new
generation recurrence rows.

### MoE FP4/INT4 identity reversal (2026-07-04)

The SM90 `NVFP4 -> Marlin` collector path introduced on the divergent
`c8c00f42` line, independently introduced on the current line by `9a5fe012`,
and made explicit in YAML by `06227023` is rejected. The framework-internal
ability to repack an NVFP4 checkpoint into a different weight-only execution
path is evidence that this is not an NVFP4 measurement; the collector does not
expose that path as a fallback or supported backend in the stock runtime. Git
author metadata on those local commits uses the release owner's identity; the
path was generated during the agent-led 0.5.14 upgrade and was not introduced
by the earlier B200 work.

The collector contract keeps three precision families distinct:

- `int4_wo` may use Marlin as its weight-only W4A16 backend;
- `nvfp4` requires a native NVFP4 backend and must fail if it resolves to
  Marlin; and
- `w4a16_mxfp4` / `w4a8_mxfp4_mxfp8` require their declared MXFP4 backend and
  must also fail if they resolve to Marlin.

Before correction, the SM90 getter emitted 113,376 unique cases: 44,955 BF16,
38,799 FP8-block, 20,493 invalid NVFP4-to-Marlin, 6,051 W4A16-MXFP4, and 3,078
INT4-WO. All 6,051 SM90 MXFP4 cases already resolve to
`flashinfer_mxfp4` (GPT-OSS 20B/120B and DeepSeek-V4-Pro); all 3,078 INT4-WO
cases are Kimi-K2.5 Marlin. The corrected SM90 target is therefore 92,883
cases with zero NVFP4 rows. Existing H20 rows labelled NVFP4 but sourced from
`sglang_marlin_moe` are rejected evidence and must not be published or reused.
Blackwell NVFP4 mappings remain separate and require their native-backend
hardware validation.

SM90 NVFP4 is absent from the retained plan; it is not an expected-failure
case. The resolver and runtime errors are defensive contract checks for a
manually constructed invalid NVFP4/MXFP4-plus-Marlin invocation, not support
for collecting that invocation.

This section supersedes the present-tense SM90 NVFP4/Marlin claims and counts
in `MOE-ONE-RUNTIME-TRUTH` and `MOE-BACKEND-ARTIFACT`. Those earlier entries
remain only as chronology for the rejected snapshot; they do not describe the
current collector contract.

#### Full-model module-loader consequence

The same audit found a second SM90 path outside the MoE benchmark itself.
`collect_mla_module.py` loaded GLM-5/5.2 and targeted DeepSeek-V3.1 NVFP4
checkpoints through `modelopt_fp4` before timing their BF16 attention modules.
SGLang 0.5.14 selects Marlin as the SM90 FP4 policy before model construction,
so the setup still enters through an unsupported NVFP4-to-Marlin contract even
when the timed module has a BF16 key. The two-layer GLM-5.2 dummy override
excludes those layers' MoE weights, so this record does not claim that its
timed DSA path executed Marlin or that Marlin changed its measured capacity.
DeepSeek-V3.1 is the direct execution proof: its NVFP4 exclusions do not cover
`o_proj`, and the timed MLA forward executes that projection through Marlin.

This is `not_applicable` for the pinned SM90 collector, not an expected
failure. Before correction, the full SM90 DSA getters emitted 264 context and
24 generation outer tasks, canonicalizing the BF16 GLM consumer key to
`nvidia/GLM-5.2-NVFP4`; the skip-indexer getters emitted another 88 context
and 8 generation tasks for that artifact. The correction filters NVFP4
checkpoints before operation-local canonicalization, so ordinary GLM DSA uses
the BF16 `zai-org/GLM-5` artifact and SM90 skip-indexer has no valid GLM-5.2
artifact. Direct loader calls fail before SGLang initialization. Native
NVFP4 module paths on SM100/103 remain distinct and must reverse-test with
unchanged getter sets; SM120 remains separately gated by its module policy.

The canonical six-getter outer-task set changes on SM90 from 404 tasks
(`06dcaad8...`) to 308 (`74222334...`): 192 old invocations are removed and
96 BF16-GLM replacements are added. Context and generation stay at 264 and 24
tasks while replacing 88 and 8 GLM-5.2 invocations respectively; skip-indexer
changes from 88/8 to 0/0; both WideEP MLA getters remain byte-for-byte equal at
10 tasks. SM100 and SM103 each remain byte-for-byte equal at 570 tasks with
canonical hash `6037fbce...`; SM120 remains the same empty module set. These
hashes cover operation-qualified serialized getter tuples from the pinned
image, not inner runtime rows.

The old H20 GLM-5.2 module rows were collected with the rejected setup and are
not publishable. This includes 17,587 full-context, 17,357 frozen
skip-indexer-context (17,567 in the reconciled diagnostic), and 1,224 rows for
each generation variant. Unaffected-model rows in those combined files remain
diagnostic evidence only; a corrected accepted artifact requires a new source
snapshot, plan fingerprint, and namespace.

The standalone GLM sparse kernels do not load a quantized model and never
initialize Marlin. Their SM90 selector nevertheless moves to the registered
BF16 `zai-org/GLM-5` identity so shapes derive from the corrected module
artifact rather than the rejected GLM-5.2 run. The newly published BF16
`zai-org/GLM-5.2` artifact is not registered or frozen in this repository and
is deliberately not added as part of this correction; onboarding it and
restoring 1M/skip-indexer coverage is a separate artifact effort.

The three sparse getters remain 33 outer tasks on SM90, but all 33 identities
move from GLM-5.2 (`f8f56c96...`) to BF16 GLM-5 (`544f1938...`). Their static
inner target changes from 13,328 to 12,100 rows. Together with ordinary DSA
changing from 17,616/1,224 ideal GLM-5.2 context/generation rows to
15,744/1,192 BF16 GLM-5 rows and skip-indexer changing from 17,616/1,224 to
zero, the explicit SM90 GLM scope decreases from 51,008 to 29,036 rows
(`-21,972`). SM100/103 sparse outer sets remain byte-for-byte equal at 33 tasks
with hash `f8f56c96...`; this reduction is not copied to native Blackwell.

### DSV4 context exact historical sets

The H20 static context plan is 20,392 rows per attention kind. The historical
source-backed live-pool omission is the following 12 `(batch_size, isl,
prefix)` shapes, repeated across two GEMM modes and four TP values for 96 rows:

```text
(512,1,1000000)  (512,1,1048575)
(512,4,1000000)  (512,8,1000000)  (512,16,1000000)
(1024,1,524288)  (1024,1,1000000)  (1024,1,1048575)
(1024,4,524288)  (1024,4,1000000)
(1024,8,524288)  (1024,8,1000000)
```

Without any other accepted gap, the maximum H20 output is therefore 20,296.
The three historical CSA outer tasks that the user explicitly deferred are
`batch_size=1024`, BF16 GEMM at TP2 and TP4, plus FP8-block GEMM at TP2. Each
lost the same 25 `isl=8` cells at prefixes
`0,1,4,8,16,32,64,128,256,512,1024,1536,2048,3072,4096,6144,8192,10000,
10240,12288,16384,32768,65536,131072,262144`. If and only if all 75 reproduce
and remain explicitly accepted, CSA is 20,221. Partial reproduction subtracts
only the exact reproduced cells. HCA has no corresponding accepted runtime
gap. These sets are an H20 replay contract, not a Blackwell skip list.

### DSV4 explicit-pool evidence boundary

There is no comparable explicit-pool last-good to restore blindly.
Clean `9a5fe012` removed the old cap but also restricted context collection to
prefix zero, so the nine nonzero-prefix workspace failures were latent rather
than directly reproducible on that checkout. The first executable combination
is the current prefix restoration plus `max_total_tokens=None`, frozen in
snapshot `3f58b147...`. This cross-links `DSV4-PREFIX-RESTORE` and
`DSV4-DERIVED-POOL-WORKSPACE`: neither change alone explains the observed
failure set, and later Blackwell work must not attribute it to `9a5fe012`
without that input-axis condition.

`50f12ed1` introduced the per-worker swept-KV cap while its framework manifest
still pinned SGLang 0.5.10, and no local hardware artifact is cryptographically
bound to that commit. The only sizeable SGLang 0.5.14 explicit-cap smoke used a
different collector predating the current DSV4 full/SWA pool rescaling
(`e86d0c4d...`), sampled only batches
1/4/16/256, skipped the current bs16 failures as `KVPoolCapacity`, omitted
batches 8/32/512/1024, and retained only 1,680/5,600 sampled CSA rows. It cannot
prove either the nine current workspace points or large-batch pool
construction. The June 28 19,001-row full run explicitly logged
`max_total_tokens=None` for all 88 workers, so it is evidence for the failing
framework-derived state, not the old cap. Any bounded-pool change is therefore
new SGLang 0.5.14 behavior requiring its own comparator and both-platform
validation, not a proven revert to a last-good implementation.

The strict H20 `None` baseline from snapshot `3f58b147...` is now immutable:
5/88 CSA context outer tasks done, 83/88 failed, and 19,073/20,392 unique
positive rows. Its exact gap split is 96 source-backed pool cells, all 75
user-deferred bs1024 cells, and 1,148 unapproved pool/workspace cells. It adds
no missing key versus June 28 and recovers exactly the old 72
`(isl=1,prefix=1,048,575,bs=1..256)` AcceleratorError cells. CSV SHA is
`6d5c18b2...` and checkpoint SHA is `90047ae6...`. Preserve this failing state
as the reverse comparator; it is not accepted full data.

The exact replay oracle is the read-only JSONL file
`/raid/aic_sglang_0514_sm90_post_b200_20260701/remaining/dsv4_modules/validation/dsv4_csa_context_strict_missing_replay_oracle.jsonl`,
SHA256 `fb4fa253b82f8012e805ae5612a5fceca12ce2a692b7d0b63f1aa61faa4500a5`.
It contains one metadata record and all 1,319 sorted ideal-minus-output keys,
including complete physical, Python-consumer, and Rust-consumer projections
and the 96/75/1,148 classifications. Independent readback proves ideal 20,392,
output 19,073, outside 0, and duplicate 0. The 83-outer-task errors JSON is
supporting evidence, not a substitute for this inner-key replay set.

The strict HCA companion is clean: 88/88 outer tasks, zero errors, and exactly
20,296 unique positive rows. Its 96-key gap is byte-for-byte the twelve
source-backed live-pool shapes across two GEMM modes and four TP values, with
no other missing or outside key. CSV SHA is `b65d7f99...`; checkpoint SHA is
`d0832d70...`. This recovers all 72 old HCA AcceleratorError cells and proves
that CSA's remaining failures are C4-indexer-specific rather than a general
DSV4 module, GPU-state, or prefix-path failure.

### DSV4 bounded-pool comparator contract

The next comparator must leave `mem_fraction_static` on the framework default
and let `DSV4PoolConfigurator` preserve all pool ratios. For a context cell,
the exact FP32 C4-logits allocation is
`W = (bs * isl) * ceil((prefix + isl) / 256) * 256` bytes. After model load but
before `alloc_memory_pool()`, reserve the maximum workspace for the worker from
SGLang's profiled pool bytes, ask the exact configurator for the resulting
token capacity, and cap it by the worker's maximum page-rounded full/SWA
requirement. This is a diagnostic candidate, not accepted product code.

It must recover the nine exact TP1/BF16 failures and preserve their nine
nearest half-workspace/prefix-262144 successes. It must also construct and run
bs512 `(prefix=524288, isl=16)` and bs1024 `(prefix=262144, isl=8)`, while the
known next pool tier remains explicitly classified, plus one TP8/FP8-block
control. Then rerun HCA context, both generation phases, and the complete CSA
exact-key diff. If one global workspace reserve would exclude an individually
feasible high-pool/low-workspace cell, group only the Pareto-distinct worker
sets; do not introduce multiple runners or a helper abstraction preemptively.
SM100 and SM103 must record profiled bytes, reserved workspace, final token
cap, actual pool sizes, and selected logits leaf independently. SM120 may use a
different leaf and cannot inherit this formula without source and hardware
proof.

Status binding as of 2026-07-03: the comparator is no longer future work.
Against frozen manifest `3f58b147...`, the three framework-default TP1/BF16
bs8/16/32 slices each reproduce the exact three target OOM/`InternalError`
cells while three nearest controls pass; an additional TP8/FP8-block reverse
slice reproduces two target errors while four controls pass.  The 400M
diagnostic A/B writes all six rows for each slice, and bounded bs512/bs1024
each write their required single row, giving bounded counts `6,6,6,1,1,6`
with zero inner errors.  Exact hashes and both rejected wrapper attempts are
preserved in `remaining/dsv4_pool_comparator/RESULTS.md`; every valid scenario
used a fresh container and returned GPU 0 clean.  This proves the mechanism,
not a product constant.  The next state is a new live-configurator-derived
candidate, which requires a new read-only snapshot before GPU acceptance.
The frozen comparator evidence manifest is `ARTIFACTS.sha256`, SHA256
`340c3c06f0f7644b67179b43e6846aee0d6dd3cfc8d773ef5a274d736b13eb89`;
all listed files verify and no evidence file/directory is writable.  Its
task-local cache subtree is explicitly excluded from evidence.

Exact-image source anchors for that decision are
`arg_groups/deepseek_v4_hook.py` (CUDA `page_size=256` and default
`swa_full_tokens_ratio=0.1`),
`model_executor/model_runner_kv_cache_mixin.py::_resolve_memory_pool_config`
(profile, configurator, then one token constraint), and
`layers/attention/dsv4/indexer.py` plus `metadata.py` (FP32 MQA-logits call
over `query_rows` by `max_c4_seq_len`, where the C4 page is 64 tokens).
SM90/100/103 currently select the CUDA DeepGEMM family; SM120 has an explicit
Torch fallback branch. This source distinction is why the product candidate
may be platform-neutral in control flow but must not claim the SM120 workspace
formula until that whole-module leaf is hardware-proven.

### GLM sparse strict full and top-k chunk contract

The immutable strict top-k comparator is snapshot `3f58b147...`, collector
source SHA256 `38a95c8aac364c72545ecad4f9411d0f9737414a98a8b3543998bb25d08861ea`.
Its CSV/checkpoint/errors SHA256 values are respectively `a041d93f...`,
`b34aa577...`, and `1e8a24e...`. The read-only exact replay oracle is
`/raid/aic_sglang_0514_sm90_post_b200_20260701/remaining/glm_sparse/validation/glm5_topk_exact_replay_oracle.json`,
SHA256 `97494a2e3062b5ab46fc6cf9f0e60c5a2ae8e6c9125403d9589add711e5d1c1e`.
It binds all 72 physical gaps and both persisted score modes to the stable
artifacts and to the exact top-k log section SHA256 `47003c2e...`; independent
readback proves ideal 3,332/6,664, actual 3,260/6,520, missing 72/144,
outside 0, duplicate 0, and no partial pair. These are replay inputs, not an
accepted skip set.

The `f4d1bcc7` causality statement is deliberately scoped to allocation:
that commit changed the selected GLM model/ceiling enumeration and thereby
made the 1M family executable, but it did not change the monolithic
`_bench_glm5_mqa` or `_bench_glm5_topk` allocation functions. It is therefore
an exposure commit, not a zero-diff claim for the whole file.

For context let `Q=bs*isl`, `S=prefix+isl`, and `K=bs*S`. The failing collector
allocates one compact FP32 score `[Q,pad4(S)]` per calibration mode. It keeps
the prior `flat` tensor alive while the `top_last` expression allocates at
least one same-size random tensor and further temporaries. The initial source
audit followed the ragged helper, which computes a query-row budget and, in one
loop, executes `fp8_mqa_logits(q[start:end])`, transforms that logits chunk,
and writes the result with concatenated width `K` and absolute row starts. A
completed review proves the configured H20 `TopkTransformMethod` selects
`PAGED` instead of that ragged transform for retained context. Compact geometry
remains the rejected old behavior, while the bounded concatenated geometry is
retained for the paged correction; only the ragged transform/artifact is rejected.

The pre-implementation correction contract therefore required the then-assumed
ragged width/offsets, one in-place score chunk, and release between score modes.
The first design
proposal also described summing the per-chunk top-k latencies as production
sequential latency. A pre-implementation review rejected that stronger claim:
the current MQA collector captures all query chunks in one CUDA graph, whereas
the proposed top-k implementation would benchmark one graph per chunk and sum
their independent steady-state kernel latencies. That sum is a **kernel-only
additive calibration**. It is not boundary-equivalent to MQA and is not the
production `MQA_i -> topk_i -> copy_i` interleaved loop. Power must not be
summed. This proposal correction happened before an execution-code edit or GPU
measurement, so there is no superseded product snapshot or data artifact.

The initial product patch was scoped to top-k, with the measurement boundary
explicit in its artifact/run record and a small input forced through
both unchunked and artificially chunked paths to record value parity and the
latency delta.
If graph-boundary effects are material relative to the recorded repeat noise,
stop and review a separate MQA/top-k benchmark-contract change instead of
silently retuning or reusing the old MQA rows. Likewise, SGLang's serving
budget includes static-memory and cached-budget state; a standalone
free-memory/30% bound must be labeled an approximation unless the relevant
0.5.14 policy is actually reproduced. The candidate was required to pass the
extreme `(16,1024,1000000)`, its three successful neighbors, the
equal-score-shape pair `(16,4096,159744)` and `(128,512,163328)`, and stress
tuple `(32,4096,1044480)` that can run only chunked. The rejected compact
`[Q,S]` proposal called it 512 GiB; corrected concatenated `[Q,bs*S]` geometry
makes the same tuple 16 TiB. That edit required a new read-only
snapshot and a fresh complete 6,664-row file; do not append repaired rows to
the failing CSV. Because the unregistered DSV3.2 wrapper delegates to this
worker, preserve its delegation/tag and decode selector contracts; a direct
actual-model long-context smoke may be recorded without registering or
publishing the path. Do not turn that adjacent debt into a new full-plan scope.
Do not alter the separately qualified DSV4 top-k calibration path unless a
shared helper actually changes.

The implemented uncommitted context state makes the independent boundary
fail-closed by passing `allow_graph_fail=False` for every chunk/mode; a row
cannot silently mix eager and CUDA-graph measurements. Decode retains its old
single-kernel `allow_graph_fail=True` path and is not relabeled as production
decode. The state also logs `Q`, concatenated `K`, rows per chunk, chunk
count, budget bytes, free bytes, and total bytes whenever it actually chunks.
This state passed 34/34 focused host contracts and is frozen read-only at
`post_glm_topk_chunk_v1`, manifest
`893724083c1f11a0944cc87d97b98ce5c604f624b1ada2661f75fd310a1658bc`.
The GLM collector moved from failing SHA256 `38a95c8a...` to
`77dd7cf361d7da657153dfe0d083922662f849d97e156fee1f173708851d7e4f`;
the DSV4 file changed only stale comments/log labels to SHA256 `e906ae1b...`.
At freeze time no GPU row had been accepted from this snapshot. The subsequent
full result is appended below; every later revert or relaxation must add a new
identity rather than replacing either the pre-run or completed state.

The bounded H20 oracle itself has an append-only revision trail. V1 script
`c2740afd...` incorrectly required exact unsorted index order and failed with
stderr SHA256 `9877674c...`; the full result already matched the sorted PyTorch
reference. V2 script `fd366241...` changed the contract to per-row set equality
and passed (`b8546919...`), proving full, chunked, and reference sets match
while the kernel's unsorted order varies with launch batch shape. V3 script
`ff0be4af...` added both score modes and all three benchmark boundaries. Its
result `e58ae847...` shows independent-graph absolute top-last latency is 9.5%
above a single-graph chunk loop, but the consumed `flat - top_last` delta
differs by only 0.1225%; the product forced-chunk delta agrees with the direct
additive delta. Thus the absolute boundary remains disclosed, while there is
no evidence-based reason to expand this patch into an MQA rewrite.

Seven fresh-container H20 candidate-path probes then passed with the frozen source:
the original `(bs=16,isl=1024,prefix=1M)` failure now executes 36 chunks; its
three long-context controls execute 9, 18, and 19 chunks; the equal local-work
pair `(16,4096,159744)` and `(128,512,163328)` executes 23 and 184 chunks; and
the logical 16-TiB score shape `(32,4096,1044480)` executes 588 chunks in
151.9 seconds. Every result has finite positive `flat`/`top_last` latency,
uses `fast_topk_transform_ragged_fused`, binds manifest `89372408...`, and
leaves no GPU process after container exit. These are smoke/contract results,
not substitutes for full collection. That later `collect.py` run was
row-complete across all
11 outer tasks in 18 minutes 52 seconds with zero errors and exactly
6,664/6,664 rows. CSV SHA256 is `e327b330...`; checkpoint SHA256 is
`da98312d...`; canonical key hash is `e714eac7...`; source counts are 6,376
ragged context and 288 paged decode rows. Structural top-k validator
`36662f4f...` and split-provenance three-op composite validator `1bd717b1...`
both return PASS, but neither audits backend selection. The completed selector
audit semantically rejects the ragged artifact and its composite.
The composite manifest `5fc2ab60...` retains old MQA/DSA snapshot identities
and the rejected ragged top-k identity separately. The semantic validator does not read
`PROVENANCE.md` or `ARTIFACTS.sha256`; source-file integrity is separately
fail-closed by `sha256sum -c ARTIFACTS.sha256`, which passes in the composite
root. Final logs contain no OOM, traceback, CUDA error, or task failure, and all
GPUs returned to 0--4 MiB/0% utilization.

### GLM top-k selector/decode audit and rejected mixed-path artifact

The blocking read-only audit is
`/raid/aic_sglang_0514_sm90_post_b200_20260701/remaining/glm_topk_backend_audit/README.md`,
SHA256 `10e5b7fd15b19f6f9dfe85799a335d9f0c00760b915dfab8eb12e5d295568500` at the
final pre-freeze review checkpoint.
It binds SGLang source `49e384ce...` and the exact selector chain:
`server_args.py:3403-3471`, `dsa_backend.py:642-656,820-824,2488-2504`,
`dsa_indexer.py:911-973`, and `dsa_topk_backend.py:88-123`.
RAGGED requires FP8 index KV, `flashmla_sparse`, and EXTEND simultaneously;
every other combination is PAGED. SM90 `auto` BF16 selects
`flashmla_sparse` but misses the FP8 predicate, while the retained FP8 profile
selects `flashmla_kv` and misses the backend predicate. Both therefore dispatch
`fast_topk_transform_fused`. H20 logs independently bind FP8
`flashmla_kv` (`af155991...`) and BF16 `flashmla_sparse` (`0e939c46...`).

`2ae0a2ae` first introduced all three execution-contract mismatches: context
`isl > 1 => ragged fused`, present-zero decode `row_starts`, and pad4 decode
score stride. `f027123f` predates the file and has zero direct GLM-file diff;
the defects are inherited/not introduced by Blackwell. Exact `topk.cu` SHA256
`f899cb9d...`, lines 488--505, proves that optional presence selects prefill and
absence plus `prefill_bs == B` selects decode. Exact `dsa_indexer.py` lines
601--620 and 704--724 prove page-size-64 logits rounding. Among 144 decode
shapes/288 rows, only 37 shapes/74 rows execute top-k; all 74 selected the wrong
leaf, 34 shapes/68 rows also had non-64 stride, and three injected nonzero
ceilings happened to be aligned. The 214 zero rows return before launch.

The append-only worktree history is: `c5cba92d...` fixed context and passed an
incomplete 36/36 suite; `8f05cbfd...` omitted decode `row_starts` and passed its
then-current 37/37 suite (`e41b8657...`); `6da3683b...` added pad64 decode; and
current pre-freeze `1194e6b6...` explicitly releases each materialized chunk
page table before the next allocation. That last change narrows peak-memory
lifetime and is not a reproduced correctness fix. Current tests are
`89026f7d...`; final host review is 50/50 plus Ruff/format/diff green. No state
has corrected GPU rows yet. Preserve concatenated context geometry, absolute
row starts, bounded Q chunks, omitted decode optionals, and pad64 decode stride.
Acceptance requires context and decode value/leaf oracles, H20 smoke with leaf
and peak-memory evidence, a new read-only snapshot, all 6,664 rows, and a
validator that rejects the old mixed-path artifact. B200 must reverse-validate
the same contracts rather than reinterpret Hopper latency.

The first H20 selector-probe execution is retained as an orchestration failure:
it observed PAGED, then exited 1 because its task-local perf-output directory
did not exist (log `f14a7d74...`, exit file `4355a46b...`). After creating only
that output directory, the unchanged probe `6ce0ff7b...` was rerun against the
same frozen `6748d7f2...` snapshot. It exits 0 with 37/37 observations of
FP8 + `flashmla_kv` + EXTEND + `sgl-kernel` => PAGED (log `1a9145a0...`,
exit file `9a271f2a...`) and one smoke row `5226bfb3...`; all GPUs are clean.
This records the failed and successful wrapper states separately. It validates
only the SM90 selector, not context/decode values and not the source-derived
SM100 `trtllm`/PAGED expectation. Blackwell must rerun the same selector probe
and append its actual backend/transform result to this chain.

The bounded H20 value gates now also pass against frozen collector
`1194e6b6...`, each in a fresh container with a clean natural exit. Context
oracle `4a433d91...` proves unchunked and forced-three-row PAGED output match a
non-identity PyTorch mapped-index reference at `(2,8,4096)`; log
`160b1f0f...`. Flat/tied full and chunk selections differ but independently
pass uniqueness/range, which is the stated tie contract. Decode oracle
`20a45ad2...` proves sequence/page width 4097, score stride 4160, omitted
`row_starts`, decode leaf present/prefill leaf absent, CUDA Graph use for both
modes, and the mixed `[2048,2049]` boundary; log `f429f874...`. These close
bounded SM90 gates only, not long-context stress or the 6,664-row rerun. Since
SM100 source selects a different backend, Blackwell must first observe its
selector and adapt the equivalent value/leaf oracle to that actual path rather
than copying the H20 `sgl-kernel` expectation.

The PAGED stress-runner also has a pre-execution reversal. Draft `53d00907...`
misreported a materialized chunk page table as `[rows, bs*S]`; review rejected
it before any GPU run because the actual table is `[rows,S]` while only the
score is `[rows,bs*S]`. Corrected runner `5b80b17a...` separates those fields,
adds baseline/peak allocator evidence and live row-budget inputs, and
fail-closes on source `1194e6b6...`, manifest `6748d7f2...`, kernel source, and
`chunks*2` mandatory graph calls. Static checks and independent read-only
review pass; no stress result existed at approval time. The 2026-07-03 resume
preflight found clean GPUs but a 100%-reported root filesystem, so all
task-owned cache/log/tmp data must remain on `/raid` or tmpfs and no unrelated
Docker cleanup is permitted. A Docker-layer allocation failure is environment
evidence, not a collector failure.

The corrected runner then passes all eight H20 stress shapes from fresh
containers. Chunk counts are 3--588 and mandatory graph calls are exactly
twice each count; all eight bind source `1194e6b6...`, manifest `6748d7f2...`,
and `fast_topk_transform_fused`, with finite positive latency and no error-log
match. The new batch-one ceiling `(1,16384,1032192)` peaks at
60,567,992,832 allocated bytes and completes in three chunks, while logical
16-TiB `(32,4096,1044480)` completes 588 chunks/1,176 graph calls in 152.734
seconds at 31,023,648,256 peak allocated bytes. Artifact identities are frozen
in `paged_stress_v1/STRESS_ARTIFACTS.sha256`; all containers exit naturally
and all GPUs return to 0--4 MiB. This closes the H20 long-context execution and
peak-lifetime gate only. Full 6,664-row recollection remains required, and none
of these H20 latency or memory values may be transferred to Blackwell.

Fresh full-run validator `930069f7...` independently reconstructs 3,188
context plus 144 decode physical keys and two score modes, for 6,664 rows. It
returns exit 2/INCOMPLETE against the empty new namespace (`3983249e...`) and
exit 1/FAIL against rejected row-complete `89372408...` data (`059d6de1...`).
That negative replay is important: the old artifact still has 11/11 tasks,
6,664 rows, and canonical key hash `e714eac7...`, but the validator rejects its
6,376 ragged context sources and known-bad CSV identity. Acceptance now
requires 6,664/6,664 `fast_topk_transform_fused` rows plus exact phase/mode,
latency, checkpoint, summary, and Docker-exit checks in a wholly new
`glm_topk_paged_full_v1` namespace.

That no-resume H20 run now completes in about 19m28s: Docker exit 0, checkpoint
11/11 done with zero failed/expected-failed, and SGLang 0.5.14 summary errors 0.
Validator `930069f7...` exits 0/PASS (`4e0a28e7...`) against all 6,664 rows:
3,188 context plus 144 decode physical keys, two exact modes each, phase rows
6,376/288, modes 3,332/3,332, and all sources
`fast_topk_transform_fused`. Canonical key hash is `e714eac7...`; new CSV
`0755831a...` is distinct from rejected ragged CSV `e327b330...` despite the
same keys. Checkpoint/summary are `6222ba81...` / `96ed4b8e...`. Raw postflight
shows the task container absent and all eight H20s at 0--4 MiB with no compute
process. This closes the SM90 PAGED top-k artifact gate. It does not close the
SM100/103/120 rows or permit any H20 latency to be relabeled as Blackwell.

Status binding as of 2026-07-03: earlier table text saying `1194e6b6...`
“remains GPU-unvalidated”, that its H20 full run “remains pending”, or that no
corrected GPU row exists is a historical statement bound to the named
pre-GPU/pre-full review checkpoint.  It is superseded, not erased, by the
accepted result above.  Current SM90 state is selector/value/stress/full
complete.  Only the separate Blackwell hardware pass and the other independently
tracked SGLang operations remain pending.  The accepted full namespace is
read-only; `ARTIFACTS.sha256` has SHA256
`414e1195a617ed735dd3b65474cd76d2eb04369bc510a6d17ab36aa97cee7ed0`
and verifies all 17 listed artifacts with no writable file or directory.

### Proposal and validator revision log

| Item | Chronology | Execution status and reason |
| --- | --- | --- |
| DSV4 bounded-pool comparator README | `0d235c44...` described only the bs8 default A/B; `cfbdb1a5...` added bs8/16/32 but accidentally dropped the explicit bs1024 control from the prose; `6c5f1796c521749abd5ae72da3eaa206a6cb7450d7be93b88ba8e4e96891fa09` restores both bs512 and bs1024 controls. | The first two designs were rejected before any GPU run. Only the final read-only README may drive the future comparator; none is product evidence. |
| GLM sparse strict validator | `b9d79e4...` allowed one paired extreme top-k gap; strict snapshot `3f58b147...` then found 72 physical/144 paired gaps; `dec88c99723c57865c371ef86ec9e174607987a94125a7d0082f57b8b55a6df8` requires both SM90 and Blackwell to have 11/11 tasks and 6,664/6,664 rows. Ragged snapshot `89372408...` uses top-k validator `36662f4f...`; split-provenance composite validator is `1bd717b1...`. | The old exception was invalidated and `dec88c997...` continues to exit 1 against its immutable 6,520-row comparator. The separately pinned validator and composite PASS structural/key checks at 6,664/6,664, but selector/decode review rejects 6,376 context rows and 74 nonzero decode rows, including 68 wrong-stride rows. A future validator must reject this row-complete mixed-path artifact. Composite provenance integrity is separately checked by passing `sha256sum -c ARTIFACTS.sha256` because the semantic validator does not read the provenance manifest. |
| GLM top-k chunk benchmark proposal | Initial prose called per-chunk sums production sequential latency; pre-implementation review corrected this to kernel-only additive calibration because MQA uses one graph around all chunks while context top-k uses one graph per chunk. The implementation also changed context from graph/eager fallback to mandatory graph. Oracle V1 rejected exact unsorted order, V2 corrected comparison to set equality, and V3 added both score modes and all benchmark boundaries. The same stress tuple changed description from 512 GiB under rejected compact `[Q,S]` geometry to 16 TiB under bounded concatenated `[Q,bs*S]` geometry. | The wording reversal predates product code/GPU data and remains labeled proposal-only. Snapshot `89372408...` is row-complete after V3 and seven H20 ragged-path probes but is semantically rejected by context selector plus decode leaf/stride review. Preserve geometry/chunking lessons only; persisted latency is mixed wrong-path evidence, kernel-only additive rather than end-to-end serving latency, and has no additive power meaning. At that review checkpoint `1194e6b6...` remained pre-freeze/GPU-unvalidated; it is now frozen and passes selector/value/eight-shape stress gates, while full 6,664-row acceptance remains pending. |

### B200 auto-heal input inventory

Execution-file diffs are not sufficient causality evidence: `f027123f` also
changed the plan inputs below. Hash comparison is against that commit's file,
not a claim that every old row is still consumed by SGLang 0.5.14.

| Input paths | Current relationship to `f027123f` | Current evidence / Blackwell obligation |
| --- | --- | --- |
| `base_ops/attention_context.yaml`, `base_ops/attention_generation.yaml` | Both are superseded in the current worktree. | Current H20 dense retained files are complete at 51,258 context, 40,468 generation, and 7,679 encoder keys. SM100/103/120 getter deltas remain pending and must be derived from these current inputs, not the auto-heal counts. |
| `base_ops/mla_module.yaml` | Byte-identical to `f027123f`, but its consumers, framework version, model set, and backend selectors changed later. | The current H20 DSA/full/skip evidence validates the exact 0.5.14 execution plan only. B200 must regenerate counts and selected leaves; byte identity does not make old B200 data current. |
| `base_ops/moe.yaml` | Superseded by the current quantization/backend matrix. | H20 focused modes pass, but full MoE is deliberately last and still pending. B200 must regenerate per-quant counts and must not revive the private auto-heal MXFP4 patch. |
| `Gemma4ForConditionalGeneration`, `Llama4ForConditionalGeneration`, `MiMoV2FlashForCausalLM`, `NemotronHForCausalLM`, `Qwen3VLForConditionalGeneration`, and `Qwen3VLMoeForConditionalGeneration` model-case YAMLs | Every listed current file differs from its `f027123f` blob; these inputs are superseded, not retained verbatim. | Per-model SGLang 0.5.14 routing is validated on the retained H20 paths described above. Later Blackwell work must record current getter deltas and actual backend leaves model by model; presence in the old auto-heal input is not proof of consumption or correctness. |

### DSV4 live-cap candidate status binding

This section appends to `DSV4-DERIVED-POOL-WORKSPACE`; it does not rewrite the
historical table row. A chronology correction is required for later replay:
`864b987d` introduced the pre-Blackwell fixed 0.7/16M policy;
`57ba6e3c` extended Blackwell support with zero pool/prefix-setting delta;
`2e197491`, not `299aaea9`, performed the R091 collector-file rename;
`1d12d321` introduced the model YAML without module prefixes; and
`299aaea9` then added and consumed its 28 module prefix values while inheriting
the fixed cap. B200 auto-heal `f027123f` has zero relevant collector,
model-YAML, and case-generator diff. `50f12ed1` added swept-KV plus 5% under
SGLang 0.5.10. Divergent `c8c00f42` and `9a5fe012` independently changed the
cap to `None`, while `9a5fe012` separately limited context to prefix zero. Only
the current prefix restore plus `None` made the workspace failure family
executable. This remains **inherited / not introduced by Blackwell**. The
live-configurator cap is new exact-0.5.14 collector integration behavior, not a
restored proven last-good and not a Blackwell-specific fix.

The earlier workspace expression was proposal shorthand and omitted the
physical DeepGEMM logits stride. Exact sgl-deep-gemm 0.1.3 uses
`split_kv=256`, so the accepted calculation is
`raw_c4=ceil((prefix+isl)/page)*(page/4)`,
`stride=align_up(raw_c4,256)`, and `W=bs*isl*stride*4`. The bs8/16 maximum is
therefore 8,204,058,624 bytes. Independent review caught both a lexicographic
maximum-selection error and the missing stride alignment before candidate GPU
acceptance. Those two states are proposal/code-review reversals with no
accepted product artifact.

The uncommitted candidate is frozen as `post_dsv4_live_cap_v1`: source-manifest
file SHA256 `0dadc469...`, collector `30b9baf9...`, contract tests
`a21041ac...`, and runner `a473624e...`. It applies only to CSA context with
input `None` and the exact-release default DeepGEMM leaf on SM90/100/103. The
model is loaded before deriving the minimum cap from live page size, SWA ratio,
profiled bytes, retained full/SWA requirements, exact logits workspace, and
the native `DSV4PoolConfigurator`. It uses the same runtime chunk and source
capacity filters and fails closed when the requirements cannot coexist.
HCA, generation, explicit diagnostic caps, and alternate leaves are unchanged.
SM120's Torch path is explicitly fail-closed and remains unsupported here.

Focused H20 evidence passes all six fresh-container scenarios with row counts
`6,6,6,1,1,6`: 26 unique positive rows, zero errors/skips, all input caps
`None`, and all selected leaves SM90 DeepGEMM. It recovers the nine exact
bs8/16/32 failures while preserving adjacent controls and also passes the
bs512, bs1024, and TP8/FP8-block boundaries. Corrected pinned-container tests
pass 22/22; the initial exit-4 package-metadata wrapper failure remains in the
bundle. The strengthened validator binds exact shapes/schema/metadata, source
identity, all derived and actual pool calculations, and the clean eight-GPU
postflight. Frozen evidence manifest SHA256 is
`2803697e0b636c6853af8095f911059c495d15bba44fdbeabf2be014dc2df88a`;
the exact historical Docker shell line and raw pre-smoke preflight were not
persisted and are disclosed rather than reconstructed.

Status binding: this accepts only the focused SM90 gate. Fresh full CSA/HCA
context and generation exact-key runs remain pending at
20,296/20,392, 20,296/20,392, 1,600/1,600, and 1,600/1,600 respectively
(43,792 aggregate). SM100/103 remain source-derived and hardware-unvalidated;
SM120 remains fail-closed. Later Blackwell work must append measured pool,
workspace, selected-leaf, and exact-key results to this same section and rerun
the SM90 reverse contract before changing the policy.

## Required update fields

Treat every Hopper/Blackwell edit as a two-sided change even when only one
platform is available. Before changing code, add or identify its ledger row;
after changing code, record both the target-platform result and the reverse
untouched-platform result. A later platform pass must update the same row
rather than replacing its earlier provenance or silently reopening the old
behavior.

Treat the evidence in each row as append-only. A later pass may refine a
classification, but it must retain the earlier failing revision, exact case,
and successful controls. Do not replace a failed Hopper or Blackwell result
with only the final green state. While a fix is uncommitted, identify it by
the affected file/path and frozen diff or snapshot hash; once authorized and
committed, add the commit without erasing the uncommitted provenance. This is
the replay checklist for the next platform, not only a summary of the current
one.

Record every add/revert/reapply cycle chronologically in the same row, even if
the eventual Git history is squashed. Each transition must say why the prior
state was rejected, which snapshot or commit contained it, and which positive
and failing controls were rerun. A final green result never erases the
intermediate behavior; this is how the later Blackwell pass can distinguish a
real platform difference from a previously repeated mistake.

For every later platform adjustment, add or update one row with:

- symptom and exact failing cases;
- first-bad or contract-dropping commit and last-good commit;
- whether the change restores a lost contract or adds new behavior;
- SM90 and SM100/103/120 getter-count deltas;
- source/probe evidence and hardware-validation status;
- positive target-platform test and reverse untouched-platform test;
- nearest same-family successful controls that any skip/guard must preserve;
- exact affected framework/collector execution path and the pre/post-fix
  artifact or snapshot identity;
- fix commit once a commit is authorized and created.

Do not remove an entry merely because one platform passes. Close it only after
the stated reverse tests and target-hardware checks are recorded.

### DSV4 full SM90 stage update (2026-07-03)

The frozen `post_dsv4_live_cap_v1` snapshot now has a fresh full CSA-context
H20/SM90 result: 88/88 outer tasks, 20,296/20,392 exact keys, precisely 96
source-capacity omissions, zero failed/expected-failed tasks, positive unique
latencies, and a clean eight-GPU postflight. This extends the focused 26-row
gate without changing the candidate source. It is H20 execution evidence, not
H200 publication data and not Blackwell evidence.

The first strict validator `5738d4a2...` rejected that otherwise complete
artifact because of three validator-envelope assumptions: child records were
looked up in aggregate Docker stdout, summary skips were treated as
capacity-only instead of chunk-plus-capacity, and eight INFO initialization
records were required to be absent. The collection was left unchanged.
Revised and independently reviewed validator `2cd2b85c...` passes the same
20,296 rows and 88 child logs; the append-only revision record is bound by
source-bundle manifest `b639b0e5...`. This is a validation-contract correction,
not an add/revert/reapply of the Hopper or Blackwell execution path. HCA context
and both generation stages remain pending at this dated checkpoint; SM100/103
remain source-derived/hardware-unvalidated and SM120 remains fail-closed.

Before accepting HCA, a second read-only audit found that `2cd2b85c...` parsed
but did not require non-CSA pool records. No HCA/generation artifact had been
accepted. Final validator `72dd57b4...` now requires one positive max/full/SWA
pool record in every worker log and preserves all CSA live-cap invariants; it
independently passed review and revalidated the unchanged CSA data. Current
source-bundle manifest is `815910a0...`; `b639b0e5...` remains historical
revision evidence. This also changes validation only, with zero execution-path
or plan-input delta on Hopper or Blackwell.

Fresh full HCA context is now also accepted on the same H20 snapshot: 88/88
tasks, 20,296/20,392 exact keys, the same precise 96 source-capacity omissions,
zero task failures, one positive pool record per worker, and clean postflight.
Final validator `72dd57b4...` passes HCA alone and both context stages together
at 40,592 rows. Current validation-bundle manifest is `51bb6bc4...`. No source,
plan, or backend selector changed between CSA and HCA; this is additional SM90
execution evidence only and does not alter the pending B200 obligations.

Both generation paths are now also accepted at 88/88 tasks and 1,600/1,600
exact keys each, with one positive pool record per worker, no runtime errors,
and clean postflight. The four-stage final validator passes 43,792 unique
positive rows. Final source-bundle SHA256 is `023bd96f...`; the 775-file
read-only evidence manifest SHA256 is
`3fbf1181702fa70b7192a075b4b631dbb4e22127a6020b84d37c0ea51754597d`,
with zero writable non-cache evidence paths. This closes full H20/SM90 DSV4
execution for snapshot `post_dsv4_live_cap_v1`. SM100/103 remain unmeasured,
SM120 remains fail-closed, and none of these H20 rows may be reused as
Blackwell or H200 performance data.

### MoE final-stage readiness (2026-07-03)

No final MoE GPU row is accepted yet. A fresh H20 namespace is frozen against
the same `post_dsv4_live_cap_v1` source manifest. Its exact SM90 stock plan is
114,105 one-task/one-row cases: 45,684 BF16, 38,799 FP8-block, 20,493 NVFP4,
6,051 W4A16-MXFP4, and 3,078 INT4-WO. The artifact validator binds the exact
checkpoint task-ID set/hash to the consumer-key set and per-key kernel source,
because the CSV does not contain model, architecture, or backend fields.

Seven family smokes plus one deterministic full-plan sample (eight cases each)
are required after all DSV4 stages end. Independent source reconstruction and
runner/validator review approve hashes `2156bebe...` (smoke), `9d7b326f...`
(full), and `9df89d74...` (validator); frozen bundle manifest is `e0546ebf...`.
This is preparation only, with no GPU/Docker execution. It covers SM90 stock
SGLang only: WideEP remains out of scope, and the recorded SM100/103 backend
distribution still requires B200 regeneration rather than reuse of this plan.

The first hardware smoke gate retained 48/64 rows. GPT-OSS MXFP4 and
DeepSeek-V4-Pro W4A16 both failed 8/8 before the kernel because TensorRT-LLM
could not create `/root/.tensorrt_llm/tmp` on the read-only container root.
Docker/collector exit 0 despite task failure exposed a runner-envelope bug;
strict validation rejected the data. Failure manifest `dca7abfe...` is frozen
and remains diagnostic, not product evidence.

Runner revision `eaa3f41f...` adds only the task-local TensorRT cache mount and
an exact 8/0/0 checkpoint gate; full runner `1a199719...` similarly requires
114,105/0/0. Validator `7a8e29fc...` fixes only the `full_sample=mixed` command
metadata false positive and continues binding exact task/key/kernel contracts.
Both failed families then pass fresh 8/8 reruns. The complete 64-key smoke gate
passes and is frozen under source manifest `f7d5cdb2...` and accepted-smoke
manifest `1309fdb9...`. This is SM90 evidence only; no backend/platform rule was
changed and Blackwell must reproduce the corresponding families independently.

### MoE SM90 full alignment failure and approved prune (2026-07-03)

Full attempt v1 processed all 114,105 tasks but is rejected: 113,376 done,
729 failed, zero expected-failed, runner exit 4, strict validator FAIL, and a
clean postflight. Every failure is Nano BF16 `flashinfer_cutlass` with
intermediate size 1856: TP16 produces local inter 116 and TP32 produces 58,
both violating the exact CUTLASS `%8` contract. The failed grid is 27 token
sizes × three distributions × nine TP/EP pairs. There was no OOM, worker
crash, restart, or environment failure. Frozen failed-attempt manifest SHA256
is `eeadf647806568e056cf5eb0902a28336222d6a0a380b947510fd3b81931ce63`.

Source and Git archaeology classify this as **current uncommitted integration
omission / not introduced by Blackwell**. Pinned SGLang selects CUTLASS for
SM90 Nano BF16 without a TP fallback, and FlashInfer requires eight-element
alignment. `c8c00f42` had matching plan/runtime guards; the current integration
retained routing and runtime rejection but omitted the plan guard. Neither the
B200 auto-heal inputs nor `7e8df6fa` introduced this omission.

The user approved skipping the uncommon TP16/32 slices. The implementation
contract is deliberately narrower than that wording: prune only BF16 cases
whose resolved backend is `flashinfer_cutlass` and whose hidden or TP-local
inter dimension is not divisible by 8. This removes exactly the 729 failed IDs
and preserves Nano TP1/2/4/8 plus same-backend Ultra TP16/32 positive controls.
The new SM90 plan target is 113,376. SM100/103 remain unaffected because Nano
uses `flashinfer_trtllm` there; SM120 is source-derived and must revalidate the
same backend-aware predicate on actual hardware.

The plan-time guard is now implemented and mirrors the runtime guard exactly.
The collector/test file SHA256 values are `8700874990...` and `4a411b6f0f...`.
Container validation passes 35 focused tests, 27 parallel-run tests, and 399
remaining collector tests; Ruff and diff checks pass. Independent read-only
review reconstructs 113,376 unique SM90 tasks, exactly equal to attempt v1's
successful set and disjoint from its 729 failures. The new sorted task-ID and
consumer-key SHA256 values are `dfa28d4d...` and `b243dd0d...`. SM100/103
resolver checks retain Nano through `flashinfer_trtllm`; SM120 applies the same
backend contract because it resolves to CUTLASS. This is implementation and
static/oracle evidence only; a new frozen snapshot and fresh H20 run remain
required, and attempt v1 stays rejected.

The corrected source is frozen as `post_moe_cutlass_align_v1`: 1,474 files,
no symlinks/write bits, manifest-file SHA256 `f22a01d0...`. Fresh namespace
`moe_cutlass_aligned_full_v2` binds runners `84076aa3...`/`cdd0600a...` and
validator `2d8df5e7...`; its 13-entry source/control manifest is `0d7df85f...`.
Static fail-closed controls pass. The new hardware gate has 96 rows: the prior
64-family matrix, Nano 24 (covering retained TP8), and Ultra 8 (covering valid
same-CUTLASS TP16/32). This records preparation only; the fresh smoke and full
H20 results remain pending, and no v1 row is reused.

Fresh v2 smoke now passes 96/96 from 00:18:00 through 00:35:46 on H20. All ten
stages have Docker/runner exit 0, zero failed/expected-failed tasks, exact
task/key hashes, and clean pre/postflight. Nano passes 24/24 including TP8;
Ultra passes 8/8 including valid TP16/32. The 113-file frozen smoke manifest
SHA256 is `80738056...`. This supersedes only the previous “smoke pending”
status for snapshot `f22a01d0...`; full 113,376/0/0 and all Blackwell hardware
obligations remain pending.

Corrected full is now accepted for snapshot `f22a01d0...`: the fresh no-resume
run completed 113,376/0/0 from 00:36:43 through 04:31:50, with 113,376 unique
positive CSV rows. Strict all-stage validation and independent reconstruction
both pass exact task SHA `dfa28d4d...`, consumer-key SHA `b243dd0d...`,
quant/backend/kernel distributions, per-key source, command/source identity,
summary/log, and clean lifecycle checks. CSV/checkpoint SHA256 values are
`3de99f97...`/`102351b8...`. No v1 checkpoint or row was resumed.

The only large diagnostic log contains SGLang's default Triton-config and
experimental NVFP4 notices; there are no failure/OOM/CUDA/restart signals or
`errors_*.json`. Final postflight is clean on all eight H20s. The 18-entry
read-only full manifest SHA256 is `0ac52de9...`. This closes the H20/SM90 stock
MoE gate for SGLang 0.5.14; SM100/103/120 hardware regeneration and reverse
validation remain required, and these H20 latencies must not be relabeled as
H200 or Blackwell data.

Reverse measurement validation is also complete. Against rejected H20 v1,
the corrected v2 artifact has the exact same 113,376 keys and sources; latency
v2/v1 is p01 0.8938, median 1.0000, p99 1.1194, with every row inside 0.5--2.
Frozen comparison manifest is `c20cfb65...`. This is direct negative evidence
that restoring the plan guard changed only enumeration, not retained Hopper
execution behavior.

For diagnostic continuity only, comparison with repository H200/SGLang 0.5.10
finds 43,497 common keys. Of those, 41,148 retain the source and 2,349 change
from Triton to CUTLASS, exactly the separately audited Nano selector change.
The source-matched H20/H200 ratio has median 1.3876, p01 0.4523, p99 3.3139,
and no <0.05 or >20 outlier. The old H200 table itself has 8,505 duplicate
consumer keys, so per-key median was used; the rejected first comparator that
assumed uniqueness is retained. Diagnostic manifest is `5eab9206...`. This
cross-device/version comparison does not authorize H200 relabeling or replace
the required Blackwell runs.

The 34 source-matched H20/H200 ratios above 10 are one FP8 Triton
`7168/2048`, top-k-8, 256-expert balanced cluster. Direct same-H20 replay
comparison across the two independent full runs gives v2/v1 min 0.9957,
median 0.9998, max 1.0083; no row is outside 0.8--1.2. Companion manifest is
`9a2d1063...`. This closes the concern as stable cross-device/version behavior
for this Hopper reverse check; it does not predict or validate Blackwell
performance.

### final_v1 collector-only candidate status binding (2026-07-04)

This appendix rebinds present-tense status without deleting the preceding
failure, correction, and comparison chronology. The final candidate execution
root is `/raid/aic_sglang_0514_sm90_final_20260704`; its clean execution-code
HEAD is `1b37712659554e834ac520a18eb1c0620a0e16df`, and the read-only
`source_final_v1/code_snapshot` manifest-file SHA256 is
`2c53ee9d73c1364c471024ca8ee94b6e5578ce7444c6c4a99f55a72f0c50f3bc`.
It uses stock SGLang 0.5.14 from image digest
`sha256:5027e95bf6ec536856b1b52a91d1f35ff5c564ab83e8a94758a169ff09bb8df3`.

The execution platform is eight NVIDIA H20-3e GPUs at SM90. `h200_sxm` is
only the existing plan selector. This bundle contains no H200, SM100, SM103,
SM120, or SM89 execution result, and no H20 row may be published or relabeled
as H200 or Blackwell data. Blackwell behavior in this candidate remains
source-derived and hardware-unvalidated.

The MoE chronology is now explicitly
`114,105 pre-alignment -> 113,376 alignment-pruned -> 92,883 final target`.
The 113,376-row CSV `3de99f97...` and manifest `0ac52de9...` passed their
then-current structural validator, but the later precision/backend identity
audit found 20,493 NVFP4 rows and every one used `sglang_marlin_moe`. Marlin
is an INT4-WO backend in this collector contract, so that entire artifact is
semantically rejected and retained only as historical diagnostic evidence.
Filtering it in place cannot create an accepted final artifact; the 92,883
target requires a fresh run from this final snapshot.

At this dated transition, fresh evidence from `final_v1` is accepted for:

- dense attention: 50,901 context, 40,468 generation, and 7,679 encoder rows;
- DSA modules: 264 context tasks / 46,688 rows and 24 generation tasks /
  3,568 rows, with no skip-indexer rows;
- GLM sparse: 3,025 MQA, 6,050 PAGED top-k, and 3,025 DSA-attention rows for
  BF16 `zai-org/GLM-5`; and
- a 64-row MoE smoke spanning BF16, FP8-block, W4A16-MXFP4, and INT4-WO,
  with Marlin only on INT4-WO and no NVFP4 row.

The three listed full stages have zero failed and zero expected-failed tasks
and pass their exact task/key/source validators. At this transition the fresh 92,883-row
MoE full run is in progress and acceptance remains pending; changing partial
progress is intentionally not recorded as evidence. A later appendix must
bind its final checkpoint, row and key counts, source/quantization contracts,
artifact hashes, and clean postflight before calling the H20/SM90 `final_v1`
bundle current.

### PR #1302 rebase transition (2026-07-04)

The preceding `final_v1` status is now historical. At the user's request, the
collector-only branch was rebased from base `9ce84ebb` onto upstream PR #1302
head `4c6cfebdc5bfbfd28f15ebad4aa9d17199f94b3a`. The first rebased SGLang
head is `28c283cd`. PR #1302 retires the selector/`sm_exceptions` mechanism,
adds positive hardware capability floors and hang-only denylisting, makes
ordinary failures observable data, and binds checkpoints to framework/SM
identity. Those are execution and population changes, not documentation-only
rules, so no row from `source_final_v1` validates the rebased product.

The in-flight `final_v1` MoE run was stopped by stopping only its task-owned
container. Its preserved rejected namespace is
`runs/moe_full_pre_pr1302_stale_20260704T211655`: 7,116/92,883 tasks done,
7,137 partial rows, zero failed/expected-failed tasks before interruption,
Docker/runner exit 137, and validation exit 1. GPUs 0--6 returned clean; the
unrelated GPU-7 process was not touched. The namespace is chronology only and
must not be resumed, filtered, finalized, or published.

Conflict resolution follows the new policy:

- `sm90_exceptions.yaml` and `sm100_exceptions.yaml` remain deleted; none of
  the old shape selectors was restored;
- expected-failure prediction, version-specific executor switches, OOM
  prediction, and per-failure cleanup from the old worker patch were dropped;
- the only retained worker delta converts an operation's successful integer
  `EXIT_CODE_RESTART` return into PR #1302's existing explicit restart path,
  preserving GDN/MoE persistence-before-recycle without changing whether a
  case runs or how failures are classified; and
- relative to PR #1302 head, the rebased branch has no direct diff under
  `collector/vllm`, `collector/trtllm`, `src`, or `rust`.

Dense, DSA, and GLM full results above remain accepted evidence only for frozen
snapshot `1b377126...`/manifest `2c53ee9d...`; they are not post-rebase passes.
Before any new GPU acceptance, the rebased state requires host tests, canonical
plan/task/key diffs under the new capability model, backend/source review,
fresh smoke, and a new read-only source manifest. Blackwell remains
hardware-unvalidated.

### Post-#1302 rule audit proposal (2026-07-04; not yet executed)

This entry records the decision boundary before changing the rebased product.
The comparison snapshot is `28c283cd`; the local backup ref is
`backup/sglang-0514-post-pr1302-pre-rule-audit-20260704`. Canonical SM90,
SM100, SM103, and SM120 plan counts and hashes are being frozen under
`post_pr1302_pre_rule_audit_plan` before implementation. Until a later entry
binds those artifacts and a new commit, every item below is a proposal only.

The audit found four SGLang-owned ways that an attempted point could disappear
or succeed without data:

- `f9c1e29c` added fixed GDN context token/value thresholds. They are not a
  live-memory feasibility calculation or a framework limit, and can make a
  grouped task succeed after silently omitting inner points. The proposed
  change removes only those thresholds; outer GDN task IDs are unchanged and
  every inner point either persists both rows or contributes to a failing
  group summary.
- `f9c1e29c` drops every Blackwell FP8-live-attention point by SM before the
  resolved backend is considered. The proposed change keeps those cases in
  the plan: SGLang 0.5.14 TRTLLM-MHA and FlashInfer paths raise rather than log
  a false FP8 live-activation row, while the Triton/FA3 paths retain their real
  FP8 input behavior. The older SM120 large-Q/O predicate is source evidence
  from the 0.5.10 Triton path, not proof for every 0.5.14 backend; it is to be
  parked as `FIXME(kernel-limit)` at invocation instead of silently filtering.
- `fa65531a`/`f9c1e29c` added MoE alignment, DSV4 FP4 EP/TP/token/SM, and
  backend-limit filters even though the invocation already raises, or can let
  the framework report the failure. These predicates are to be removed from
  generation. Stable artifact `allowed_modes`, universal MoE mathematics,
  and physical-key dedup remain. The shared Qwen `max_tp_exclusive` declaration
  is intentionally unchanged because changing it here would alter TRT-LLM and
  vLLM population; the user's low-priority TP16/32 coverage decision remains a
  separate shared-plan review.
- the inherited `37826f10` SM120 DeepGEMM-module early return is a broad getter
  skip. Stock DSA is to use the #1302 registry maturity marker on hardware that
  has not been validated, while the getter itself remains executable for a
  future targeted bring-up. GLM-5.2 skip-indexer is likewise explicit at the
  registry on SM90, where the only registered checkpoint is NVFP4 and is
  removed by the positive SM100 capability floor.

This proposal changes no vLLM, TRT-LLM, SDK, Rust, packaged-data, executor,
case-ID, or failure-classification behavior. SM90 execution remains the reverse
gate; SM100 is retained for the dependent B200 continuation. SM103 and SM120
remain source-derived/hardware-unvalidated and are not advertised as supported
by this transition. Exact before/after invocation and key-set deltas, focused
tests, and the resulting product identity must be appended before any new
collection artifact is accepted.

### Post-#1302 rule-audit implementation checkpoint (2026-07-04; uncommitted, no GPU evidence)

Upstream PR #1302 still resolves to `4c6cfebdc5bfbfd28f15ebad4aa9d17199f94b3a`,
which is an ancestor of rebased head `28c283cd`. The current worktree has no
frozen product identity yet. No GPU row produced by this worktree is accepted.

The collector-only changes that do not alter executor or case-generation
mechanisms are now implemented for review:

- standalone SM90 MLA generation no longer applies the old FlashMLA int32
  bound to FA3 cases;
- DSA and DSV4 decode graph benchmarks enter SGLang 0.5.14
  `model_capture_mode`, selecting the same dual-/multi-stream branch as serving;
- GEMM, encoder attention, compute-scale, MLA-BMM, dense attention, MLA, mHC,
  GDN, and MoE persistence paths fail when `log_perf` does not write a row;
- GLM/DSV4 sparse benchmarks no longer fall back silently from CUDA Graph to
  eager, and grouped failures retain each inner shape plus its original error;
- GEMM, MLA-BMM, and mHC rows name their invoked SGLang/Torch/TileLang path
  instead of the generic `sglang`/`default` label; and
- DSA module rows derive a composite source from the initialized framework
  backend (`use_mha`, `dsa_prefill_impl`, or `dsa_decode_impl`), distinguishing
  dense FA3/TRTLLM-ragged, full indexer, and true skip-indexer paths instead of
  repeating the configured outer `dsa` label; and
- framework-layer MoE validates the constructed quant method/runner before
  timing and derives provenance from that object. In particular, DSV4
  `w4a16_mxfp4` on SM90 currently constructs `Fp8MoEMethod` under the generic
  Triton request, so it now fails before timing instead of persisting FP8 work
  under an MXFP4 label. GPT-OSS SM90 MXFP4 remains the real Triton MXFP4 path;
  Kimi INT4-WO remains Marlin; SM100/103 NVFP4 and MXFP4 remain source-derived
  FlashInfer paths pending their own hardware validation.

Focused exact-image tests pass 58/58 after these changes. Worktree-wide
non-parallel and fresh-process parallel checks pass 386/386 and 26/26 on the
same current files. Ruff and `git diff --check` also pass.

The audit also found a mechanism boundary that this checkpoint deliberately
does not cross. DSA context has 264 grouped outer tasks: its worker currently
reduces 70,224 structurally admitted inner candidates to 46,688 attempts using
a fixed `2^25` condition and runtime chunk/pool filters. DSV4 CSA and HCA each
reduce 56,936 inner candidates to 20,296, mostly through the serving chunk
size. Moving model/phase structure and live-memory feasibility before queueing
requires a deterministic grouped-inner manifest/task fingerprint; representing
long prompts by actual serving chunks changes the benchmark boundary. PR #1302
classifies both as mechanism decisions requiring explicit owner approval. Until
that decision is recorded, do not run or claim complete DSA/DSV4 context data,
do not convert the filters into YAML/capability/denylist rules, and do not copy
Hopper chunk or capacity values to Blackwell.

### DSV4 context hidden-state feasibility checkpoint (2026-07-04; uncommitted, no GPU execution)

The release owner has now approved one narrow part of the grouped-inner
mechanism: filter a DSV4 context inner shape before queueing when the mandatory
BF16 input hidden state alone exceeds the live device-capacity budget.  The
exact lower bound is `batch_size * sequence_length * hidden_size * 2` bytes, where
`hidden_size` comes from the selected model config and the budget is 80% of the
smallest visible CUDA device's reported total memory.  Equality is retained.
The remaining 20% is reserved for the loaded layer, KV pools, graph state, and
other mandatory tensors.  This is deliberately not a fixed H20, SM90, model,
sequence, chunk, or free-memory threshold.

The source batch/sequence axes remain owned by
`DeepseekV4ForCausalLM_cases.yaml`.  Grouped DSV4 module collection and its
inner sequence expansion originate in `299aaea9`; the current SGLang 0.5.14
chunk/pool alignment is carried by `f9c1e29c` and `cfd933a7`.  Those serving
chunk and live KV-pool decisions are outside this change.  The implementation
must attach each getter-retained inner sequence manifest to the outer task so
the scheduler ID and resume checkpoint change exactly when the executable
inner set changes.  It must log raw, retained, and dropped expanded-inner
counts together with the device budget and formula.

Scope is stock SGLang 0.5.14 DSV4 CSA/HCA context only.  Generation allocates
one hidden row per request rather than `batch_size * sequence_length` rows and
is unchanged.  DSA, GLM sparse, MoE, WideEP, vLLM, TRT-LLM, SDK/Rust consumers,
database keys, and packaged data are unchanged.  The same capacity-derived
rule may run on SM90/100/103/120, but only the platform on which the getter is
executed supplies the budget; no H20 capacity result may be copied to
Blackwell.  Canonical before/after task and expanded-inner counts/hashes,
boundary tests, untouched-framework reverse checks, and the final commit must
be appended before accepting GPU data from this transition.

The uncommitted implementation now attaches a canonical
`((prefix, (sequence_lengths...)), ...)` manifest to every retained context
outer task.  A private pure function in `case_generator.py` owns the unchanged
model-position admission; the SGLang getter applies only the memory predicate
afterward.  The subprocess consumes that manifest instead of rebuilding the
source/model grid.  On the recorded eight-H20-3e capacity of
150,110,011,392 bytes
per device, the 80% budget is 120,088,009,113 bytes.  An exact-image static
reconstruction gives the following; this is plan evidence only because another
task currently owns the GPUs:

| Op | Before outer/task SHA | After outer/task SHA | Before expanded/hash | After expanded/hash |
| --- | --- | --- | --- | --- |
| CSA context | 88 / `2b1f7fad...` | 88 / `8e6be3c8...` | 56,936 / `a239748e...` | 52,432 / `06e27170...` |
| HCA context | 88 / `55b4d7bf...` | 88 / `b5c7fad6...` | 56,936 / `e12d0c52...` | 52,432 / `60823950...` |

Each context op therefore drops 4,504 expanded inner cells (7.91%) before
queueing.  The canonical Flash-FP8 batch-1024 prefix-zero manifest retains
sequence lengths through 12,288 and omits 16,384 and above; the result is
derived from its config `hidden_size=4096`, not from a sequence constant.
Before the memory predicate, the YAML axes expand to 61,600 cells and the
pre-existing max-position rule structurally admits 56,936, disclosing 4,664
structural omissions separately from the 4,504 memory drops.
Sorted unique task IDs and expanded `(outer-case, prefix, sequence)` records,
each LF-terminated, define the hashes above.

CSA/HCA generation remain 88 tasks with unchanged hashes `74358772...` and
`1c71cc62...`.  No shared YAML input or central public getter output changed,
and no vLLM/TRT-LLM file changed.
Context smoke generation now binds its own 352-cell manifest instead of
shrinking a full manifest in the worker: CSA/HCA smoke task hashes are
`c6a17d19...` and `9585e225...`.  This fixes context smoke/full resume identity;
generation's older worker-expanded smoke identity remains outside this filter
change and must use a separate checkpoint namespace.
Focused no-GPU exact-image tests pass 75/75, including equality-at-budget,
minimum-visible-device capacity, deterministic task-identity change, direct
manifest consumption, and generation isolation.  The complete non-fork
Collector suite passes 390/390 and the fresh-process parallel suite passes
26/26 in the same image.  Existing runtime
chunk/KV-pool `continue` paths remain separate debt and are not reclassified as
this filter; the manifest is therefore the exact getter-retained,
memory-admitted set, not yet a claim that every cell produces a row.  The
manual CLI path also retains its historical worker-side grid and is outside
this registry/getter guarantee.  Any old DSV4 context checkpoint has a
different task identity and must not be resumed into the new plan; use a fresh
namespace because the executor does not migrate or discard the old failed IDs.
On 2026-07-05, a live getter probe in the pinned image on one H20-3e reported
SM90 and 150,110,011,392 bytes, then reproduced 88 outer tasks, 52,432 retained
inner cells, and a batch-1024 prefix-zero ceiling of 12,288 for both CSA and
HCA.  The immediately following benchmark smoke is rejected: its preflight was
clean, but an unrelated vLLM container started afterward and occupied all eight
GPUs, including the selected GPU 0.  Only the task-owned SGLang container was
stopped; the external container was untouched, and the partial row is not
accepted or resumable.  The release owner explicitly allowed the code commit
to proceed without waiting for another smoke.  An uncontaminated smoke in a
fresh namespace remains required before an accepted full collection begins.

### DSV4-Pro w4a16 non-support decision (2026-07-05)

Symptom: after `65505157` aligned the shared `w4a16_mxfp4` SM90 backend map to
the GPT-OSS truth (`90: triton`), every enumerated
`deepseek-ai/DeepSeek-V4-Pro` `w4a16_mxfp4` case — exactly 3,078 on SM90 —
constructed plain `Fp8MoEMethod` under the Triton request (pinned
`fp8.py:284-305`, `is_fp4_experts=True`) and failed closed before timing.
The parent state `28c283cd` resolved those cases to `flashinfer_mxfp4`; the
FP4/INT4 identity-reversal count above (6,051 SM90 MXFP4 including DSV4-Pro)
therefore no longer described the plan after `65505157`.

Decision (release owner, 2026-07-05): do not support the native-FP4 Pro
artifact as `w4a16_mxfp4`.  This is new deliberate behavior, not a restored
contract: the mode is removed from the artifact's sglang `allowed_modes` in
`DeepseekV4ForCausalLM_cases.yaml`, matching the trtllm row that was already
w4a8-only and the SM90 NVFP4 treatment (absent from plan, not
expected-failure).  Hopper deployments of DSV4-Pro model via the converted
`sgl-project/DeepSeek-V4-Pro-FP8` artifact.  The pinned image does provide a
native SM90 MXFP4 path — `--moe-runner-backend flashinfer_mxfp4` constructs
`Mxfp4FlashinferCutlassMoEMethod` (FlashInfer #3084 cutlass mixed-input;
serving default remains Marlin, which the identity contract rejects) — so
restoration is a per-model
`sglang_moe_backends: w4a16_mxfp4: {90: flashinfer_mxfp4}` pin plus re-adding
the mode; the YAML comment at the artifact row records that path.

Exact-image getter-count deltas (static enumeration, no GPU run required for
this plan-only change): SM90 110,970 -> 107,892 (-3,078, all DSV4-Pro
`w4a16_mxfp4`); SM100 and SM103 each 148,149 -> 145,071 (-3,078, the same
artifact's `w4a16_mxfp4` slice); SM120 unchanged (MoE parked by registry
maturity markers).  Reverse untouched controls, re-measured in the same
image: GPT-OSS `w4a16_mxfp4` stays 5,751 on SM90/100/103; native Pro
`w4a8_mxfp4_mxfp8` stays 3,078 on SM100/103 with `flashinfer_mxfp4`; converted
Pro-FP8 `fp8_block` stays 3,078 on all three; DSV4-Flash rows are unchanged
(native Flash was already w4a8-only, SM90-absent by declaration).
`test_dsv4_moe_quantization_policy_prunes_unrelated_modes` locks the
w4a8-only sglang policy.  Only case generation is affected; no collector
execution path, trtllm/vllm row, or SDK/Rust consumer changes.  The decision
lands in the same commit as this entry.

### B200 continuation pass, first hardware round (2026-07-05)

Execution context: a 4x B200 node (SM100, `total_memory` 191,495,471,104
bytes per device) running the pinned runtime directly; sglang reports
`0.5.14` and the in-image source checkout is exactly `49e384ce`. Branch base
for this pass is `604f5608`. Host baseline before any change: 391 + 26
collector tests, Ruff check/format, and `git diff --check` all pass.
Environment caveat recorded for replay: a login shell without `CUDA_HOME`
makes TileLang JIT resolve the pip `nvidia/cu13` toolchain, whose nvcc lacks
the CUDA-13 `include/cccl` headers, so every sm_100a TileLang compile fails
with `fatal error: cuda/atomic: No such file or directory`. B200 mHC (and any
future TileLang-leaf) runs must export `CUDA_HOME=/usr/local/cuda`; DeepGEMM,
FlashInfer, and Triton JIT paths did not need it.

Static SM100 enumeration on this node matches every recorded count: MoE
145,071 (the post-`604f5608` plan), DSA context/generation 264/24, skip
variants 88/8, GLM-5 sparse 11 per op, GDN 16, mHC 4, DSV4 88 outer tasks per
op. The DSV4 context memory filter derived its budget from this platform
(80% of 191,495,471,104 = 153,196,376,883 bytes; structural 56,936, retained
53,712, memory-dropped 3,224), confirming no H20 capacity value is copied.
`dsv4_hca_attn_module` and `dsv4_csa_attn_module` enumerate zero cases in
this image because `flash_mla` is not installed; that gate returns an
unlogged empty list on SM90 and SM100 alike and is recorded here as shared
population-visibility debt, not fixed in this pass.

Row updates (positive B200 evidence; the SM90 reverse gate for each code
change below is the host contract suite — 392 + 26 after this pass — plus
the explicit no-op arguments given per row; H20 hardware reruns remain owned
by the SM90 side):

- `ATTN-PER-MODEL-ROUTING` / `ATTN-VARIANT-KEY-0514`: B200 execution probes
  ran the declared SM100 routes end to end — default MHA (Qwen3) and Gemma-4
  / GPT-OSS / Llama-4 on `trtllm_mha` (sinks, window 128/1024/8192 rows
  persisted), MiMo-V2 dense+SWA on `triton` with QK192/V128 and FP8 KV,
  Qwen3.5 on `triton`, NemotronH on `flashinfer`, Kimi refused at plan level.
  Source audit: `server_args._get_default_attn_backend` returns `trtllm_mha`
  on `is_sm100_supported()` (major 10, so SM103 mirrors SM100), the Qwen3.5
  arch hook defaults hybrid-GDN to `triton` on SM100, and
  `apply_nemotron_h_defaults` sets `flashinfer`; a direct B200 probe of
  `trtllm_mha` with QK192/V128 fails in-kernel
  (`shape '[-1, 64, 1, 192]' is invalid`), proving the MiMo Triton pin is the
  only runnable 0.5.14 path. One collector defect found and fixed: the layer
  call passed `sinks=None` unconditionally while SGLang 0.5.14
  `FlashInferAttnBackend.forward_extend/forward_decode` accept no `sinks`
  kwarg, so every flashinfer-routed dense case failed with TypeError (first
  seen: NemotronH SM100 smoke, 6/8 errors). The fix passes `sinks` only when
  `has_attention_sink`, mirroring the serving call sites; SM90 is unaffected
  because its dense routes (fa3/triton) accept `sinks=None` with identical
  semantics. The AST contract test now locks the conditional-kwargs form.
  Post-fix smokes: NemotronH 6 rows + 2 expected FP8 fail-closed raises; all
  other models unchanged. The FP8 live-activation raises fired exactly where
  designed (trtllm_mha/flashinfer prefill) and were classified.

- `MOE-ONE-RUNTIME-TRUTH` / `MOE-BACKEND-ARTIFACT`: high-level FusedMoE
  smokes on B200, all with verified constructed-method provenance —
  DSV3 fp8_block, GLM-5.2 NVFP4, Nemotron-Super NVFP4, Qwen3.5 bf16+fp8_block
  via `sglang_flashinfer_trtllm_moe`; GPT-OSS w4a16/w4a8 via `Mxfp4MoEMethod`
  `trtllm_sm100` leaf (`sglang_flashinfer_trtllm_moe`); DSV4-Pro w4a8 via
  `Mxfp4FlashinferTrtllmMoEMethod` (`sglang_mxfp4_flashinfer_trtllm_moe`);
  Gemma-4 BF16 GELU on raw Triton. Kimi-K2.5 INT4 enumerates zero SM100 MoE
  cases by declaration (`int4_wo` `max_sm_exclusive: 100`), preserving the
  zero-population contract. No `f027123f` private patch was revived.

- `DSV4-W4A16-ALIGN` (extended to the W4A8 sibling): the requested B200
  TP regeneration ran TP1..32 for native Pro (inter 3,072) and Flash (inter
  2,048) w4a8. local_inter 3,072/1,536/768/384/128 pass; 192 fails inside the
  TRTLLM-gen batched GEMM (`getValidConfigIndices: No valid config`), and
  96/64 fail the FlashInfer weight-shuffle `assert M % 128 == 0`
  (`Mxfp4FlashinferTrtllmMoEMethod.process_weights_after_loading`). A new
  classified guard raises for SM100/103 `w4a8_mxfp4_mxfp8` +
  `flashinfer_mxfp4` when hidden or local_inter is not 128-aligned — same
  layer and form as the existing SM90 W4A16 guard, cases stay in the plan,
  and the raise cites the framework sites; SM103 is marked source-derived.
  Unit test `test_runtime_rejects_misaligned_sm100_dsv4_w4a8_case` locks it.
  SM90 W4A16 guard and its test are untouched.

- `GEMMA-GELU-ALIGN`: the SM100 vector-width-16 predicate fired live on B200
  (`local_inter_size=44` rejected with the width-16 message; a valid BF16
  case persisted on raw Triton). Width selection by `sm_version >= 100` is
  now hardware-exercised on SM100; SM120 remains source-derived.

- `DSV4-B200-SPARSE-PREFILL-CHUNK` (code change, ledger-directed): the
  ordered B200 probe ran. This node derives `chunked_prefill_size=16384`;
  the old collector bound evaluated to 11,622. A task-local uncapped run and
  then the committed collector both executed bs=1 prefix=0 CSA context cells
  of 8,192/12,288/16,384 fresh tokens with zero errors (latencies
  3.48/5.19/7.10 ms). Decisive source fact: stock 0.5.14 defaults
  `SGLANG_OPT_USE_JIT_INDEXER_METADATA=True` (`environ.py:789`; only HIP
  disables it), so `PagedIndexerMetadata` builds JIT metadata for every
  prefill — the instrumented probe observed `metadata_path=jit` even at
  8,192 — and with the env off the framework still switches to JIT above its
  own 11,673 threshold. The deep_gemm sched-meta SMEM formula therefore
  modeled a kernel production never exposes prefill to; the collector cap is
  removed and `_effective_prefill_chunk_size` now returns SGLang's serving
  chunk. SM90 no-op proof: H20 derives 8,192, already below the old bound,
  so plan, pool derivation, and worker skips are numerically identical; only
  chunk-above-11,622 platforms change. The `ChunkedPrefillSize` skip
  mechanism itself is unchanged and remains the separately tracked debt.

- `DSV4-DECODE-MULTISTREAM-PARITY`: the requested B200 boundary probe ran
  under the collector's real `model_capture_mode` decode path with MQALayer
  branch instrumentation: batches 64 and 128 took
  `_forward_prepare_multi_stream` (6/6 invocations), batch 129 took the
  sequential `_forward_prepare` (6/6), matching
  `_multi_stream_bs_limit = 128 if is_blackwell_supported()`. All three runs
  persisted rows with zero errors. The SM90 64/65 reverse boundary remains
  owned by the H20 side.

- DSV4 module coverage: unpatched-collector B200 cells persisted for CSA
  context (8,192/12,288/16,384), HCA context (8,192/16,384 at prefix 0 and
  4,096 at prefix 512), CSA generation (bs 64/128/129), and HCA generation
  (bs 8), all zero-error. `dsv4_paged_mqa_logits_module` enumerates 11 tasks;
  `dsv4_csa_topk_calib` 11.

- `DSA-SUBBACKEND-SELECTOR` / `DSA-PREFILL-GRAPH-BOUNDARY`: B200 smokes of
  the full DSA module ops completed with zero errors and composite
  provenance: generation rows are uniformly `sglang_dsa_indexer_trtllm`
  (TRTLLM-GEN is the runnable SM100 decode bucket, as the source predicted),
  context rows split `sglang_dsa_dense_mha_trtllm_ragged` /
  `sglang_dsa_indexer_flashmla_sparse` / `sglang_dsa_indexer_trtllm`. The
  skip-indexer ops produced their `sglang_dsa_skip_*` analogues on SM100 —
  the first hardware exercise of the `next_skip_topk` reuse path on the only
  registry-supported platform.

- `DSV32-SM100-REDUCED-HEAD-DECODE`: the four boundary families ran on B200
  in the real pipeline (DeepSeek-V3.2, TP2 -> 64 local heads and TP4 -> 32,
  kv steps 128 and 256, BF16 and FP8 KV, batches through 1,024): 4/4 outer
  tasks done, 592 rows, zero errors, all `sglang_dsa_indexer_trtllm`. The
  old `f027123f` SM100 exclusion (`head_num <= 32` and `kv_len >= 256`) does
  not reproduce on the exact 0.5.14 selector; the add-then-remove cycle
  closes on the removed side for this runtime. SM103/SM120 remain separate.

- `GLM-SPARSE-EXTREME` / `GLM-MQA-RAGGED-PAGED-BOUNDARY`: B200 smokes of
  `glm5_mqa_logits_module` (32 rows, `deep_gemm.fp8_mqa_logits`),
  `glm5_topk_module` (64 rows, `fast_topk_transform_fused`), and
  `glm5_dsa_attn_module` (32 rows, `flash_mla_sparse_fwd`) completed with
  zero errors. This is smoke-depth evidence only; the SM100 paged-selector,
  stride, and lifetime audit at scale stays open until the SM90 top-k
  recollection closes, and the 4 GiB fused-accessor ordered sweeps
  (`DSA-FUSED-KS-4G-OFFSET`, `GLM52-*-CEILING`) remain full-campaign work
  that this pass deliberately did not run.

- `MHC-PRENORM-ENV` (code change): the collector now prints the RESOLVED
  `SGLANG_OPT_USE_TILELANG_MHC_PRE/POST` and `SGLANG_OPT_DEEPGEMM_HC_PRENORM`
  values next to the raw environment. First B200 evidence:
  `resolved_deepgemm_hc_prenorm=True` (with `tilelang_pre=1`), closing this
  row's "log did not print the resolved value" gap on the Blackwell side.

- `MHC-TEARDOWN-AND-PARTIAL` (code change): the B200 smoke exposed that H20
  never validated two mHC tasks through ONE worker (its 4 tasks spread over
  8 GPU workers). On a single-GPU worker, task N+1 failed ModelRunner init
  with "Process group ... is not initialized in the world group map" because
  teardown destroyed the torch group but left `parallel_state._WORLD` set,
  and then with the `set_global_expert_location_metadata` assert because
  that module global also survives. Fixes, mirroring SGLang's own
  `cleanup_dist_env_and_memory`: teardown now calls `destroy_model_parallel`
  plus `destroy_distributed_environment` and returns
  `expert_location._global_expert_location_metadata` to its pre-init state;
  the model load moved inside the guarded region so a mid-init failure still
  reaches teardown (the old `del model_runner` in `finally` raised
  UnboundLocalError and masked cleanup when load failed). Teardown errors
  still propagate and fail the worker. Post-fix B200 smoke: 4/4 tasks,
  140/140 rows (the H20-equivalent full count), both
  `sglang_tilelang_mhc_pre` and `sglang_tilelang_mhc_post`. The H20 result
  is not invalidated (its per-task fresh workers never hit this), but the
  SM90 full-campaign rerun should confirm same-worker sequencing once.

- `GDN-DECODE-BS1024` and `WORKER-FATAL-RECYCLE`: a full B200 GDN run for
  Qwen3.5-122B-A10B recorded the actual SM100 selector — Triton
  `fused_recurrent_gated_delta_rule_packed_decode`, not a FlashInfer
  auto-selection, for this config — and reproduced the exact H20 failure:
  `batch_size=1024: Triton Error [CUDA]: invalid argument` (the grid-Y
  boundary), classified and disclosed, generation otherwise ok=10. No skip
  added. The same run validated the fatal-recycle contract in the real
  pipeline: the context task hit a fatal CUDA error, was recorded failed,
  the worker exited to reset the GPU context, and a fresh worker completed
  the following generation task with exact done/failed accounting.

- NEW observation `GDN-CONTEXT-256K-SM100`: B200 GDN context fails with
  `Triton Error [CUDA]: an illegal memory access` at total fresh tokens
  262,144 (2^18) and above — (bs=8, sl=32768), (16, 16384), (16, 32768),
  (64, 32768) all fail in fresh `CUDA_LAUNCH_BLOCKING=1` single-shape
  processes, and the failure precedes the first `causal_conv1d_fn` row, so
  the conv1d prefill leaf (not the chunk scan) is the failing component.
  Nearest successes: (8, 16384) and (4, 32768), both 131,072 tokens, persist
  conv1d + `chunk_gated_delta_rule` rows. H20 persisted all GDN context rows
  for the same grid, so this is SM100-specific framework-kernel behavior in
  the pinned image. Per failure doctrine it stays in the classified failure
  log — no denylist entry (it does not hang), no capability floor, no skip —
  and it needs an upstream-facing minimal repro plus SM103 comparison before
  any collector change is justified.

Deferred to the full B200 campaign: complete ordered DSA/GLM ceiling sweeps
(stock and safe-offset comparators), the accepted-complete CSA decision,
GLM top-k scale audit after the SM90 rerun, full MoE (still deliberately
last), and fresh-namespace full collections for every op family. No GPU data
from this pass is published as campaign data; all artifacts live under the
node-local probe directory and this entry records the durable facts.

### Affected-op full collection on B200 (2026-07-05, post-review head e849c8c5)

After the PR review round (all threads addressed and resolved), the ops whose
execution path changed in this continuation ran ONE full B200 collection from
head `e849c8c5` in fresh namespace `b200_full_20260705` (same node/image as
the probe round; artifacts are node-local evidence, not published data):

- `mhc_module`: 4/4 tasks, 140/140 rows (`sglang_tilelang_mhc_pre/post`),
  zero errors — the first full mHC pass that sequences multiple tasks through
  one worker, validating the completed teardown on hardware. The resolved
  env line printed `resolved_deepgemm_hc_prenorm=True`.
- MoE, native DSV4 w4a8 slices: Flash 2,835 rows + 324 failures, Pro 2,349
  rows + 729 failures; every persisted row carries the verified
  `sglang_mxfp4_flashinfer_trtllm_moe` provenance and every failure is the
  new classified 128-alignment guard, exactly at the probed boundary
  (Flash tp=32 only; Pro tp=16 [405] and tp=32 [324]). Zero unexpected
  failures.
- Dense attention: context 42,255 rows (trtllm_mha 30,185 / triton 9,486 /
  flashinfer 2,584), generation 46,936 (34,746 / 8,620 / 3,570). 16,408
  errors, of which 16,389 are the designed FP8 live-activation fail-closed
  raises (trtllm_mha 15,097, flashinfer 1,292, all context). The remaining
  19 are one cluster: Gemma-4 SWA profiles (head_dim 256, window 1024) on
  trtllm_mha with 1/2/4 local heads (TP32/64 shards), both phases,
  asynchronous illegal access. Fresh blocking single-shape reruns of
  representative failing cells PASS, so this is a configuration-dependent
  within-sweep state defect (the same class as the H20 GLM/DSA
  ordered-sweep findings), not a standalone shape limit; the failures stay
  classified, no skip, and the ordered-sweep reverse test transfers to this
  row for both platforms.
- DSV4 context: HCA completed 88/88 with 23,728 unique rows and zero errors
  — the first complete B200 HCA context artifact. CSA completed 48/88
  (batches 1..32) with 21,896 unique rows; all 40 failed outer tasks are
  batches >= 64 (x tp 1/2/4/8 x bf16/fp8_block), each dying at its first
  attempted cell that combines the sparse-prefill branch with a large paged
  span. Worker-side ChunkedPrefillSize/KVPoolCapacity skips remain the
  separately tracked mechanism debt (logged, 54,961 lines across both ops).

CSA root cause (isolated, deterministic, exact-image source): a cell enters
SGLang 0.5.14 `_forward_prefill_sparse` when fresh tokens exceed
`_LARGE_INDEXER_QUERY_THRESHOLD` = 11,673. That branch dequantizes the whole
c4-compressed span (`cache.c4_flat_token_ids`, deepseek_v4_backend.py:1534)
through `dequantize_k_cache_paged` (dsv4/dequant_k_cache.py), whose Triton
kernel computes `out_row_base = token_id * output_stride_0` from the int32
`tl.program_id(0)` (the page-table `loc` is cast `.to(tl.int64)`, the row
base is not). With out rows of 512 bf16 elements, offsets overflow int32
once the dequant span crosses ~2^22 c4 tokens (~2^24 raw KV tokens),
producing the illegal access. Fresh single-cell blocking probes on a clean
GPU: (bs=128, prefix=131,072, sl=96 -> fresh 12,288) FAILS deterministically;
(sl=91 -> fresh 11,648, below the branch threshold) PASSES; (sl=90) PASSES;
(prefix=0, sl=96) PASSES; earlier bs=1 prefix-0 cells at 12,288/16,384
PASSED. So the boundary is (fresh > 11,673) AND (dequant span above the
int32/stride limit) — a stock-framework defect that B200 serving (derived
chunk 16,384, long context) would also hit; H20's derived 8,192 chunk never
enters the branch, which is why SM90 never saw it. The chunk-cap removal is
therefore NOT reverted: it admits exactly what production admits, and the
failing cells are classified observations. Upstream-facing minimal repro and
an int64-cast fix suggestion (mirror the `loc` cast on `token_id`) are the
follow-up; a collector-side classified guard would need the exact
`n_compressed` predicate from the framework and is deferred to that report.
No skip, denylist entry, or capability floor is added. Reverse notes: any
future fix must rerun the B200 cells above plus the passing controls, and
the SM90 side needs no action (branch unreachable at its chunk) beyond
keeping the 8,192-chunk fact recorded.
