# Task V1: vLLM collector verification 0.24.0 → 0.27.1 — Report

Plan: `~/Vault/oakhaven-aic-support/plans/2026-08-21-aic1782-qwen38-max-vllm.md`
Branch: `yimingl/aic-1782-qwen38-max-vllm`, stacked on #1573 (base `86f07df2`).
Method: shallow `git clone --depth 1 --branch <tag>` of `vllm-project/vllm` at v0.24.0 (`ee0da84ab9e04ac7610e28580af62c365e898389`) and v0.27.1 (`6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`) into the scratchpad; every surface below was verified by reading the actual source at both tags (Read/Grep/diff), not by memory or inference. No GPU was available in this environment, so no kernel was actually executed — "verified" below means "the framework-facing surface this collector touches was read at both pinned versions and found compatible/incompatible", matching the sibling sglang round's standard (AIC-1762 Task 4c/4d).

**Status: no BLOCKED items.** All three modules needed only mechanical fixes (2 relocations/renames, both resolved via version-conditional imports); no surface required a genuine design decision.

## collect_gdn.py

Target lane: the whole op (context + generation phases), per the plan's "gdn" blanket must-verify designation.

| Surface | 0.24.0 | 0.27.1 | Class |
|---|---|---|---|
| `fused_recurrent_gated_delta_rule_packed_decode` import | `vllm.model_executor.layers.fla.ops` | `vllm.third_party.flash_linear_attention.ops` (whole `fla` package relocated; function body byte-identical) | RELOCATED-MECHANICAL |
| `causal_conv1d_fn`/`causal_conv1d_update` import | `vllm.model_executor.layers.mamba.ops.causal_conv1d` | same path | UNCHANGED — signature byte-identical (`causal_conv1d_fn` full-diff match); `causal_conv1d_update` gains one new trailing optional `out=None` kwarg, not passed here |
| `ChunkGatedDeltaRule` import + construction | `mamba/gdn/qwen_gdn_linear_attn.py:291` | same path, `:213` | UNCHANGED — `__init__`/`forward_cuda`/`forward_native`/`forward_cutedsl` bodies byte-identical (full-range diff, exit 0) |
| `.gdn_prefill_backend` / `_resolve_gdn_prefill_backend` | `qwen_gdn_linear_attn.py:150-211` | `:85-133` | UNCHANGED for this collector — same `linear_key_head_dim`-keyed flashinfer/cutedsl/triton selection; 0.24.0 additionally gates the SM100+ flashinfer branch behind `_is_libs_cu13_install_intact()` (an upstream nvidia-cutlass-dsl packaging-bug hash-check vLLM itself removed by 0.27.1, cutlass#3170/#3259) — an environment-dependent nuance, not a code change, since the collector reads `.gdn_prefill_backend` dynamically for `kernel_source` either way |
| `GDNAttentionMetadata` construction | `v1/attention/backends/gdn_attn.py:42-79` | same lines, byte-identical | UNCHANGED — all 10 fields this collector passes (`num_prefills`, `num_prefill_tokens`, `num_decodes`, `num_decode_tokens`, `num_spec_decodes`, `num_spec_decode_tokens`, `num_actual_tokens`, `nums_dict`, `batch_ptr`, `token_chunk_offset_ptr`) confirmed present with unchanged defaults by direct field-list read, not name-grep |
| `compute_causal_conv1d_metadata` | `v1/attention/backends/utils.py:810-856` | `:838-884` | UNCHANGED — full-body diff exit 0 |
| grid-y CUDA limit (`grid = (NV, B * HV)`) | `fla/ops/fused_recurrent.py:449` | `third_party/flash_linear_attention/ops/fused_recurrent.py:449` | UNCHANGED — same line, same formula, whole function byte-identical |
| `VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE` | `envs.py:115`, default `True` | `envs.py:124`, default `True` | UNCHANGED |
| `mamba_ssm_dtype=float32` Qwen3.5 pin | `model_executor/models/config.py:608-625` | `:804-821` | UNCHANGED — same warning text, same logic |
| SM120 IMA FIXME chain (`chunk.py:61 -> chunk_delta_h.py -> index.py:34 prepare_chunk_offsets`) | `chunk_delta_h.py:347` | `chunk_delta_h.py:349` (2-line shift) | citation paths updated; underlying bug NOT re-tested on hardware at 0.27.1 (no GPU here) |

**Fix applied:** version-conditional import (try 0.27.1 path first, fall back to 0.24.0):
```python
try:
    from vllm.third_party.flash_linear_attention.ops import (
        fused_recurrent_gated_delta_rule_packed_decode,
    )
except ImportError:
    from vllm.model_executor.layers.fla.ops import (
        fused_recurrent_gated_delta_rule_packed_decode,
    )
```
`__compat__` bumped: `"vllm==0.24.0"` → `"vllm>=0.24.0,<=0.27.1,!=0.25.0,!=0.25.1,!=0.26.0,!=0.27.0"`.

Tests: `tests/unit/collector/vllm/test_collector_import_surface.py::TestCollectGdnImportSurface` (4 tests: clean import at both shapes; red-proof against `PRE_FIX_COMMIT` 0.27.1-shaped fake raises `ImportError`; companion proves the pre-fix code was fine at 0.24.0-shaped fake).

## collect_moe.py

Target lanes (must-verify): `bfloat16`, `fp8_block`, `nvfp4` — the three quant modes Qwen3.8-Max's published checkpoints use. Non-target lanes (`fp8` per-tensor, `w4a16_mxfp4`, `w4a8_mxfp4_mxfp8`, `int4_wo` — serving Nemotron/Kimi-K3/DeepSeek-V4/gpt-oss, not Qwen3.8-Max) were checked at existence/shape depth only, per the plan's guard-or-verify allowance.

| Surface | 0.24.0 | 0.27.1 | Class |
|---|---|---|---|
| `FusedMoE` import | `fused_moe/layer.py:103`, module-level **factory function** returning `MoERunner`, explicitly flagged `# TODO: rename this` at `:102` | renamed to `FusedMoEFactory` (`layer.py:99`, TODO comment removed; no back-compat alias in the module or `fused_moe/__init__.py`) | **INCOMPATIBLE, fixed** — this is the one real break found |
| `FusedMoE`/`FusedMoEFactory` kwarg contract | full kwarg list read (`num_experts` … `apply_routed_scale_to_output`) | identical kwargs retained; 0.27.1 only *adds* new optional/defaulted params (`intermediate_pad`, `activation_situ_beta`, `activation_situ_linear_beta` promoted to first-class kwargs, `reduce_results`, `ckpt_names`) | UNCHANGED for every kwarg this collector passes |
| workspace/forward-context plumbing: `init_workspace_manager`, `get_forward_context`/`set_forward_context`, `.moe_layer_index` | `v1/worker/workspace.py:216`; `forward_context.py:178` | same line `:216`; `:186` (shift only) | UNCHANGED |
| backend-selection read (`unquantized_backend`/`fp8_backend`/`nvfp4_backend`/`mxfp4_backend`/`wna16_backend`) | `unquantized_fused_moe_method.py:53` `self.unquantized_backend, self.experts_cls = select_unquantized_moe_backend(...)`; `quantization/fp8.py:527` `self.fp8_backend, ...`; `quantization/modelopt.py:1410` `self.nvfp4_backend, ...` | same attribute-assignment pattern present, line-shifted only (`fp8.py:517`, `modelopt.py:1402`) | UNCHANGED — these remain plain instance attributes on the constructed `quant_method`, NOT refactored into a singleton/registry the way sglang's `MOE_RUNNER_BACKEND` moved to `RuntimeContext`/`Flags` |
| `Fp8Config.__init__` (fp8_block lane) | `quantization/fp8.py` | same | UNCHANGED — full-body diff exit 0 |
| `CompressedTensorsConfig.from_config` (nvfp4 lane) | `compressed_tensors.py:228` | `:230` | UNCHANGED signature/dict-shape parsing; `compressed_tensors_moe_w4a4_nvfp4.py` (the nvfp4 method class) near-byte-identical (only diff: two new internal `layer=layer`/`backend=self.nvfp4_backend` kwargs passed to an already-verified-unchanged internal call, not touched by this collector) |
| `FallbackExperts` / `._select_experts_impl` | `experts/fallback.py:14,127,153` | same lines `:14,127,153` | UNCHANGED — file byte-identical |
| weight-population loop (dtype-keyed `named_parameters` fill) | generic by-dtype logic | same quant methods create the same parameter dtypes for the 3 target lanes | UNCHANGED (no create_weights signature change found for Fp8Config/CompressedTensorsConfig nvfp4 path) |
| non-target lanes: `ModelOptFp8Config`, `Mxfp4Config`, `vllm.models.deepseek_v4.quant_config.DeepseekV4FP8Config`, `CutlassExpertsMxfp4._supports_current_device` | all present | all present — `DeepseekV4FP8Config` class at identical line 29 both versions; `CutlassExpertsMxfp4._supports_current_device` at identical 4 line numbers (301/693/998/1313) both versions; `self.wna16_backend`/`self.mxfp4_backend` attribute-assignment sites confirmed present at 0.27.1 (file `compressed_tensors_moe_wna16_marlin.py` was consolidated into `compressed_tensors_moe_wna16.py`, but `CompressedTensorsWNA16MoEMethod.wna16_backend` still exists there) | existence/shape-checked only; no incompatibility found, so **no fail-closed guard added** (documented decision, see `__compat__` comment) |

**Fix applied:** version-conditional import inside `run_moe_torch`:
```python
try:
    from vllm.model_executor.layers.fused_moe.layer import FusedMoEFactory as FusedMoE
except ImportError:
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
```
`__compat__` bumped the same way as GDN. No fail-closed guard was added for the 4 non-target lanes — existence/attribute-assignment checks found no incompatibility, and the guard mechanism (`_raise_if_unverified_moe_lane`, the sglang precedent) was judged unnecessary rather than added defensively with no supporting evidence of breakage. This is a discretionary call the plan explicitly allows ("others may be guarded", not "must"); flagging here for the record in case a reviewer wants it added anyway.

Tests: `tests/unit/collector/vllm/test_collector_import_surface.py::TestCollectMoeImportSurface` (4 tests). Notable technique: since `run_moe_torch`'s vLLM imports are function-local (not module-level), the test proves the full 9-statement vLLM import preamble resolved by calling `run_moe_torch(moe_tp_size=2, moe_ep_size=2, ...)` and asserting it raises the collector's own post-import `ValueError("...does not combine logical TP and EP")` — reaching that line proves every import above it succeeded, without needing a GPU or a full FusedMoE construction.

## collect_gemm.py

Target: the whole op (bfloat16/fp8/fp8_block/nvfp4), per the plan's blanket "gemm" must-verify designation.

**Zero code changes needed.** Every surface this collector touches is byte-identical or a pure line-shift between the two pinned tags:

| Surface | 0.24.0 | 0.27.1 |
|---|---|---|
| `vllm.envs.VLLM_BATCH_INVARIANT` | `envs.py:88`, default `False` | `envs.py:89`, default `False` |
| `vllm._custom_ops.scaled_fp4_quant` | `_custom_ops.py:1580` | `:1523` — signature byte-identical |
| `RowParallelLinear.__init__` | `linear.py:1493-1560` | `:1613-1680` — full-body diff exit 0, every kwarg this collector passes (`input_size`, `output_size`, `bias`, `skip_bias_add`, `params_dtype`, `quant_config`, `prefix`, `return_bias`, `disable_tp`) unchanged |
| **Core dispatch**: `FlashInferFp8DeepGEMMDynamicBlockScaledKernel` (`.base`/`.fallback`, M<32-FlashInfer/M≥32-DeepGEMM split) | `kernels/linear/scaled_mm/flashinfer.py` (338 lines) | same file, **byte-identical in its entirety** (`diff` exit 0) |
| `Fp8Config.__init__` | `quantization/fp8.py` | same — byte-identical |
| `CompressedTensorsConfig.from_config` | `compressed_tensors.py:228` | `:230` |
| `vllm.utils.deep_gemm.per_block_cast_to_fp8` | `:662` | `:662` (same line) |
| SM120 FIXME citations (`platforms/cuda.py support_deep_gemm`, `utils/deep_gemm.py should_use_deepgemm_for_fp8_linear`) | `cuda.py:663`; `deep_gemm.py:700` | `cuda.py:665`; `deep_gemm.py:700` (unchanged) — DeepGEMM/CUTLASS-side citations (`csrc/apis/layout.hpp`, `cutlass_gemm_caller.cuh`) live outside the vllm clone and were not re-derived |

This is the **opposite outcome from the sglang gemm precedent** (AIC-1762 Task 4c/4d), which found the same call shape silently dispatching to a different kernel at the newer pin — here, nothing moved.

`__compat__` bumped the same way; no import or dispatch code changed. Tests: `tests/unit/collector/vllm/test_collector_import_surface.py::TestCollectGemmImportSurface` (2 tests — forward-looking regression guard only, no red/pre-fix reproduction needed since nothing broke).

## `collector/vllm/utils.py` — transitive dependency, independently checked

Not one of the three target files, but a real framework-facing surface these collectors transitively reach: `collect_gemm.py` imports `setup_distributed`/`with_exit_stack` from it at **module level** (so `import collector.vllm.collect_gemm` alone exercises `utils.py`'s own imports), and `collect_moe.py` imports `setup_distributed` from it **inside `run_moe_torch`**. `utils.py` does substantial module-level `vllm.*` importing of its own (`vllm._custom_ops`, nine names from `vllm.config`, `vllm.config.model.ModelDType`, `vllm.distributed.init_distributed_environment`, `vllm.distributed.parallel_state.ensure_model_parallel_initialized`, `vllm.utils.math_utils.cdiv`, `vllm.utils.torch_utils.{STR_DTYPE_TO_TORCH_DTYPE,kv_cache_dtype_str_to_dtype}`, `vllm.v1.attention.backends.registry.AttentionBackendEnum`, `vllm.v1.attention.backends.utils.CommonAttentionMetadata`, `vllm.v1.kv_cache_interface.{FullAttentionSpec,SlidingWindowSpec,get_kv_quant_mode}`) — none of which either module-specific report section above traces, since both stayed scoped to their own file's direct imports. Read against both clones directly:

| Surface | 0.24.0 | 0.27.1 | Class |
|---|---|---|---|
| `vllm.config` re-exports (`CacheConfig`, `CompilationConfig`, `DeviceConfig`, `LoadConfig`, `ModelConfig`, `ParallelConfig`, `SchedulerConfig`, `VllmConfig`, `set_current_vllm_config`) | package `__init__.py` re-exports all 9 | same | UNCHANGED — `vllm.config` was already a package (not a bare `.py`) at 0.24.0 |
| `vllm.config.model.ModelDType` | `config/model.py:98`, `Literal[...]` type alias | same line | UNCHANGED |
| `vllm.distributed.init_distributed_environment` | defined in `distributed/parallel_state.py:1536`, re-exported via `distributed/__init__.py`'s `from .parallel_state import *` | moved to `parallel_state.py:1588` (52-line shift), same wildcard re-export mechanism, unchanged `__init__.py` | UNCHANGED — resolves identically at both versions |
| `vllm.distributed.parallel_state.ensure_model_parallel_initialized` | `parallel_state.py:1938` | `parallel_state.py:1992` (54-line shift) | UNCHANGED |
| `vllm.utils.math_utils.cdiv` | `math_utils.py:10` | same | UNCHANGED |
| `vllm.utils.torch_utils.{STR_DTYPE_TO_TORCH_DTYPE,kv_cache_dtype_str_to_dtype}` | present | present (`kv_cache_dtype_str_to_dtype` at `:395` @0.27.1) | UNCHANGED |
| `vllm.v1.attention.backends.registry.AttentionBackendEnum` | `registry.py:34`, class def | same | UNCHANGED |
| `vllm.v1.attention.backends.utils.CommonAttentionMetadata` | re-imported into `backends/utils.py` from `v1/attention/backend.py` (actual `@dataclass` definition there, `:394` @0.24.0) | same re-import pattern, definition now at `v1/attention/backend.py:412` (18-line shift) | UNCHANGED — resolves via the same import binding at both versions |
| `vllm.v1.kv_cache_interface.{FullAttentionSpec,SlidingWindowSpec,get_kv_quant_mode}` | all three present | all three present | UNCHANGED |

No incompatibility found; `utils.py` needed no changes. Two of these lookups (`init_distributed_environment`, `CommonAttentionMetadata`) initially grepped as absent from their expected file directly — both turned out to be wildcard/binding re-exports rather than direct definitions, resolved by reading the actual `__init__.py`/import statements rather than trusting a single grep miss (the same "read the whole file, don't just grep" discipline the sglang round's `RuntimeContext` near-miss established).

## `__compat__` grammar verification

`">=0.24.0,<=0.27.1,!=0.25.0,!=0.25.1,!=0.26.0,!=0.27.0"` parses and evaluates correctly under `version_resolver._check_compat` (all clauses are ANDed `packaging.specifiers.Specifier` comparisons — confirmed by reading `version_resolver.py`, not assumed). Same honest caveat as the sglang precedent: `!=X.Y.Z` excludes only the literal point version, so e.g. `0.25.0.post1` or `0.26.0rc1` still satisfy the specifier alone — `tests/unit/collector/test_version_resolver.py::TestExactVersionSetVllm` codifies this explicitly (`test_excluded_version_variants_leak_through_the_specifier_alone`, 5 parametrized cases) rather than asserting a false invariant. The framework_manifest digest-pinned model pin is the true version enforcement upstream and only ever supplies exactly `"0.24.0"` or `"0.27.1"` in a sanctioned run, so the leak is unreachable there. Also verified the floor/ceiling asymmetry (`test_endpoint_post_release_asymmetry`): `>=` is a floor (post-releases of it still satisfy), `<=` is a ceiling (post-releases of it do not) — same comparator semantics as the sglang class, not a new bug.

## framework_manifest.yaml

Added `frameworks.vllm.models` section, mirroring `frameworks.sglang.models` (AIC-1762 Task 4b) — same precedence semantics (model pin overrides family/default resolution for every op in the run when `--model-path` matches a key exactly; no match falls through to ordinary default resolution unchanged).

Image variants: the existing `frameworks.vllm.default` entry uses `default`+`cu129` variant keys. Queried the Docker Hub registry API directly (anonymous bearer token, `Docker-Content-Digest` response header — not the tag name alone) for `vllm/vllm-openai`:

| Tag | Result |
|---|---|
| `v0.27.1` | 200, digest `sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967` |
| `v0.27.1-cu129` | 200, digest `sha256:07913e94a58a4e61322c88f8d4647411b4dd394838b53abacaa32448d9d8de0a` |
| `v0.27.1-cu130` | **404** |
| `v0.27.1-cu128` | **404** |

Only `default`+`cu129` exist for v0.27.1 (matching the existing manifest's variant convention), so both — and only both — are pinned for all three Max ids:

```yaml
models:
  "Qwen/Qwen3.8-2.4T-A95B":
    version: "0.27.1"
    images:
      default: "vllm/vllm-openai:v0.27.1@sha256:0a51ea5b..."
      cu129: "vllm/vllm-openai:v0.27.1-cu129@sha256:07913e94..."
  "Qwen/Qwen3.8-2.4T-A95B-FP8": { ... same ... }
  "RadixArk/Qwen3.8-2.4T-A95B-NVFP4": { ... same ... }
```

Why v0.27.1 (not the 0.24.0 default) is required at all: vLLM's model registry only maps the bare `Qwen3_5MoeForCausalLM` architecture string to a loadable model class starting at v0.27.1 (`model_executor/models/registry.py:202`, verified via source clone). The model class itself already exists at 0.24.0 (`qwen3_5.py:543` @0.24.0, `:426` @0.27.1) but is unreachable through standard architecture dispatch there — 0.24.0's registry only maps `Qwen3_5MoeForConditionalGeneration` (the VLM variant) and `Qwen3_5MoeMTP` (`registry.py:641`), not the bare CausalLM key.

`test_manifest_exposes_current_framework_versions_and_images`-style coverage added: `test_model_pin_match_resolves_qwen38_max_to_vllm_0_27_1` (3 parametrized model ids), `test_vllm_model_pin_mismatch_error_names_the_model_scoped_image`, `test_vllm_unknown_model_id_falls_back_to_default_resolution` in `tests/unit/collector/test_framework_manifest.py`.

## `test_active_cuda_vllm_collectors_are_exactly_pinned_to_manifest_version` fix

This pre-existing test asserted `__compat__` as an exact `f'__compat__ = "vllm=={version}"'` string match per module — incompatible with a 2-version-set range now that moe/gdn/gemm carry one. Converted the invariant from a literal string match to a semantic check (`_check_compat(declared, resolved_version)`, the same function `collect.py`/`fullnode.py` use at runtime), which fully subsumes the old exact-equality behavior for every other vLLM module (an `==X.Y.Z` pin only ever accepts that one version) while correctly handling the three bumped ranges. Added a companion test (`test_vllm_target_lane_collectors_declare_the_exact_bumped_compat_range`) that locks the literal expected string for the 3 modules specifically, so the semantic test alone wouldn't silently tolerate an accidental widening/narrowing of the range.

## Gates

```
$ .venv/bin/python -m pytest tests/unit/collector/ -q
======================= 915 passed, 6 skipped in 28.89s =======================

$ .venv/bin/python tools/perf_database/check_collector_data.py
[OK]   R1 sidecar coverage
[OK]   R2 reuse validity
[OK]   R3 comm exclusion
[OK]   R4 family placement
[OK]   R5 identity (manifest v2 resolution)
[OK]   R6 no legacy markers
[OK]   R7 attested case plan
collector data check OK: all rules passed

$ .venv/bin/python tools/perf_database/backend_facts.py --check
OK: collector/op_backend_facts.yaml matches aic-core/src/aiconfigurator_core/systems/data (3,355 fact slices)

$ .venv/bin/python -m ruff check collector/ tests/unit/collector/
All checks passed!

$ .venv/bin/python -m ruff format --check collector/ tests/unit/collector/
157 files already formatted
```

6 skips are pre-existing GPU-gated tests, unrelated to this change (confirmed by running the same suite gate cleanly before this task's edits, per the branch's own history).

## Files changed

- `collector/vllm/collect_gdn.py` — `__compat__` bump + honest verification comment; version-conditional import for `fused_recurrent_gated_delta_rule_packed_decode`; FIXME citation update.
- `collector/vllm/collect_moe.py` — `__compat__` bump + honest verification comment; version-conditional import for `FusedMoE`/`FusedMoEFactory`.
- `collector/vllm/collect_gemm.py` — `__compat__` bump + honest verification comment (no code change).
- `collector/framework_manifest.yaml` — `frameworks.vllm.models` section (3 Max ids, digest-pinned `default`+`cu129`).
- `tests/unit/collector/test_framework_manifest.py` — semantic `__compat__` check + literal-range lock test; vLLM model-pin resolution/mismatch/fallback tests.
- `tests/unit/collector/test_version_resolver.py` — `TestExactVersionSetVllm` (vLLM analogue of the sglang `TestExactVersionSet`).
- `tests/unit/collector/vllm/test_collector_import_surface.py` (new) — strict `ModuleType`-fake import-surface tests for all three modules, red-proven against the pre-task commit for the two modules that actually needed a fix (moe, gdn).

## Concerns for the reviewer

1. **MoE non-target-lane guard decision** (see table above): 4 quant lanes not deeply verified, no fail-closed guard added because no incompatibility was found at existence/shape depth. Reasonable per the plan's discretion, but flagging since the plan's language ("fail-closed version guard" for unverified citations) could also be read more strictly.
2. **GDN SM120 IMA and GEMM SM120 block-fp8 FIXMEs**: neither was re-tested on real hardware at 0.27.1 (no GPU in this environment) — citations were re-verified as still-findable in source, not as still-reproducing.
3. Everything else in this report is either a byte-identical/line-shifted match (independently confirmed via `diff` on the actual cloned tags) or a mechanical fix with a red-proven regression test.
