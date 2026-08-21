# Task V2: vLLM case backend maps — Report

Plan: `~/Vault/oakhaven-aic-support/plans/2026-08-21-aic1782-qwen38-max-vllm.md`
Branch: `yimingl/aic-1782-qwen38-max-vllm`, stacked on #1573, base of this task 907e8ff6 (Task V1).
Method: every dispatch claim below was read from the actual `git clone --depth 1 --branch v0.27.1`/`v0.24.0` clones in the scratchpad (`vllm-0271` @ `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`, `vllm-0240` @ `ee0da84ab9e04ac7610e28580af62c365e898389`, both re-verified via `git describe`/`git rev-parse` at the start of this task), via `grep -n`/full-file reads, never inferred or copied from the sglang precedent or the dynamo recipe README. Four research forks traced the vLLM source in parallel (MoE bf16/fp8_block; MoE nvfp4; GDN; attention); I independently re-verified every decisive citation myself against the clones before writing it into the case YAML, and caught and corrected one error a fork/comment draft made (see "Process note" below).

## 0. Governing finding: vLLM has no case-schema backend-map field at all

Before tracing dispatch, I checked whether the schema mechanism the task assumes (`vllm_moe_backends`, `vllm_backends`, a GDN equivalent) actually exists. It does not, for any op:

- `case_generator.py`'s attention-profile builder has exactly two backend branches, `if backend == "sglang":` (consumes `sglang_backends`) and `elif backend == "trtllm":` (consumes `trtllm_attn_backend`) — no `elif backend == "vllm":` exists. `kernel_source` stays `None` for every vllm attention case, which is correct: the population key `(num_heads, num_kv_heads, head_dim, window_size, kernel_source)` collapses all SMs into one case per shape for vllm, exactly because vllm's dispatch is resolved dynamically at collection time, not predicted per-SM in the plan.
- MoE: only `sglang_moe_backends` is a real, consumed field (`get_sglang_moe_backend`, used by `collector/sglang/collect_moe.py:329` to **manually pin** the runner backend sglang constructs — sglang needs this because its own auto-dispatch sometimes needs an explicit choice). `base_ops/moe.yaml`'s `moe_vllm.quantization_modes` has no `backends:` map at all (unlike `moe_sglang`).
- GDN: no `*_backends` field exists anywhere in `cases/base_ops/gdn.yaml` or the model case files, for any backend, sglang included.
- A repo-wide grep (`grep -rn "vllm_backends\|vllm_moe_backends\|vllm_attn_backend\|vllm_gdn"` across `collector/`, `tests/unit/collector/`) returns zero real hits.
- I read `Qwen3_5MoeForConditionalGeneration_cases.yaml` (the 397B file) in full: it does **not** carry a `vllm_backends`/`vllm_moe_backends` key either. Its `framework_quantization.vllm.allowed_modes` entries are quant-mode gates, not backend/kernel maps. **The plan's "schema precedent exists on 397B rows" claim does not hold** — I verified this rather than assuming the plan was right, per the campaign's standing lesson about copied-not-traced claims.

Why this is correct, not a gap: `collector/vllm/collect_moe.py`, `collect_attn.py`, and `collect_gdn.py` all construct through vLLM's own builders/selectors with no explicit backend override, then **read back** whichever backend the framework itself picked (`collect_moe.py:728-739` probes `unquantized_backend`/`fp8_backend`/`nvfp4_backend`/`mxfp4_backend`/`wna16_backend` off the constructed `quant_method`; `collect_attn.py:138-145` calls `current_platform.get_attn_backend_cls(...)`; `collect_gdn.py:161` reads `chunk_gdn.gdn_prefill_backend`). This is exactly layer_permissions.md's preferred mechanism ("prefer the framework's own dispatch path"), which needs no YAML declaration and would be actively misleading to add (a `vllm_moe_backends` key would be dead, unconsumed YAML — case_authoring.md's "no new YAML keys" boundary, and a mechanism change to make it real would need explicit human approval, out of this task's scope).

**Deliverable, per item, given this finding:** citation-dense comments in `collector/cases/models/Qwen3_5MoeForCausalLM_cases.yaml`, mirroring the sglang rows' documentation density, instead of a new consumed field. Every trace below is also the comment now in that file.

## 1. MOE base row (Qwen/Qwen3.8-2.4T-A95B + FP8 alias) — bf16/fp8_block dispatch

Checkpoint identity confirmed from the packaged config (`aic-core/src/aiconfigurator_core/model_configs/Qwen--Qwen3.8-2.4T-A95B-FP8_config.json`): `quantization_config = {"quant_method": "fp8", "activation_scheme": "dynamic", "weight_per_tensor": false, "weight_block_size": [128, 128]}` — vLLM's **native** `Fp8Config` block-fp8 path, not compressed-tensors. `collect_moe.py`'s `fp8_block` branch (`Fp8Config(is_checkpoint_fp8_serialized=True, activation_scheme="dynamic", weight_block_size=[128,128])`) matches this exactly.

### bfloat16 (`quant_config=None` → `UnquantizedFusedMoEMethod`)

`select_unquantized_moe_backend` (`fused_moe/oracle/unquantized.py:194-311`). Priority list on CUDA (`_get_priority_backends:48-99`) = `[FLASHINFER_TRTLLM, FLASHINFER_CUTLASS, TRITON, BATCHED_TRITON]`; SM90-only reorder (`:77-79`) demotes both FlashInfer entries to the back ("FlashInfer unquantized MoE kernels are slower than Triton on Hopper"). A `dp_size>1` demotion of FLASHINFER_CUTLASS exists (`:81-85`) but is inert here — this collector always passes `dp_size=1` (`collect_moe.py:663`) — and is **not architecture-keyed** despite the comment naming Qwen3.5 as the discovering case (the guard is a plain `dp_size` check).

| SM | Resolved backend | Gate |
|---|---|---|
| 90 | **TRITON** | `TritonExperts._supports_current_device` (`experts/triton_moe.py:96-97`) is unconditional (`is_cuda_alike()`); FlashInfer entries demoted to the back at SM90 |
| 100/103 | **FLASHINFER_TRTLLM** | `TrtLlmBf16ExpertsMonolithic._supports_current_device` = `is_device_capability_family(100)` (`experts/trtllm_bf16_moe.py:61-69`, family = capability major 10, so both SM100 and SM103); `_supports_activation` allows SILU; `_supports_parallel_config` passes for a single-rank/no-EP collector |
| 120 | **FLASHINFER_CUTLASS** | `FlashInferExperts._supports_current_device` includes `is_device_capability_family(120)` (`experts/flashinfer_cutlass_moe.py:129-140`); its `_supports_quant_scheme` for `(None,None)` needs only `has_device_capability(90)` — a **floor**, true at 120 too (`platforms/interface.py:433-454`, the `has_device_capability` classmethod) |

### fp8_block (`Fp8Config(...weight_block_size=[128,128], activation_scheme="dynamic")`)

`select_fp8_moe_backend` (`fused_moe/oracle/fp8.py:271-419`), called from `Fp8MoEMethod.__init__` (`fp8.py:517-522`) with `weight_key=kFp8Static128BlockSym, activation_key=kFp8Dynamic128Sym, allow_vllm_cutlass=False` — `allow_vllm_cutlass=False` removes the VLLM_CUTLASS candidates (`oracle/fp8.py:390-392`). Priority baseline (`_get_priority_backends:69-133`) = `[AITER, FLASHINFER_TRTLLM, FLASHINFER_CUTLASS, DEEPGEMM, VLLM_CUTLASS, TRITON, MARLIN, HUMMING, ...]`; the only reorder for this exact `(weight_key, activation_key)` pair fires at an **exact** SM90 match (`is_device_capability(90)`, not a family/floor check — `:113-122`): `ep_size>1` → FLASHINFER_CUTLASS to front, else → TRITON to front (this single-rank collector is always `ep_size<=1` in the TP sweep, `ep>1` in the EP sweep).

| SM | Resolved backend | Gate |
|---|---|---|
| 90 | **TRITON** (TP) / **FLASHINFER_CUTLASS** (EP, ep_size>1) | exact-SM90 reorder, `oracle/fp8.py:113-122` |
| 100/103 | **FLASHINFER_TRTLLM** | `TrtLlmFp8ExpertsMonolithic`: family-100 device gate (`experts/trtllm_fp8_moe.py:96-102`) + explicit `(kFp8Static128BlockSym, kFp8Dynamic128Sym)` quant-scheme match (`:295-305`) + routing-method match for this model's Renormalize routing (`:314-330`) — **this is exactly the dynamo recipe README's flashinfer_trtllm claim, confirmed true at this SM family only, not universally** |
| 120 | **DEEPGEMM** oracle label | FLASHINFER_TRTLLM fails the family-100 gate; FLASHINFER_CUTLASS's fp8-block quant-scheme gate is an exact SM90 match — the `(kFp8Static128BlockSym, kFp8Dynamic128Sym)` branch guarded by `is_device_capability(90)` at `flashinfer_cutlass_moe.py:167-174` (NOT the `(None,None)`/`has_device_capability` floor branch a few lines above it, `:161-164` — that one is a floor, true at 120 too, but doesn't cover the fp8-block scheme) — so it fails too; falls to `TritonOrDeepGemmExperts` (a `FallbackExperts` wrapping `DeepGemmExperts`+`TritonExperts`, `experts/triton_deep_gemm_moe.py:24-38`) — `DeepGemmExperts._supports_current_device` = `VLLM_USE_DEEP_GEMM` (default `True`) and `has_deep_gemm()` and `current_platform.support_deep_gemm()` (SM90/family-100/family-120, `platforms/cuda.py:665-671`) — the actual leaf kernel is chosen per-shape at runtime, exactly what `collect_moe.py:760-771` already introspects into `kernel_source` |

One residual, explicitly-flagged uncertainty (does not change the table): `TrtLlmFp8ExpertsMonolithic._supports_router_logits_dtype` requires bf16/float32 router logits (`oracle/fp8.py:311-313`); this collector passes `router_logits_dtype=None` for this model (no routing bias, softmax scoring). **RESOLVED in the whole-branch review**: `collect_moe.py:577-578` sets `vllm_config.model_config.dtype = torch.bfloat16`; `layer.py:331` reads `moe_in_dtype = vllm_config.model_config.dtype` and passes it as `in_dtype` into `FusedMoEConfig` (`layer.py:346`); `FusedMoEConfig`'s `__post_init__`-equivalent (`config.py:1342-1343`) defaults `router_logits_dtype` to `in_dtype` whenever it's `None` — so the gate receives `bfloat16` unconditionally for this model; no config path can resolve `float16` there. The table above is unaffected either way.

Both oracle ranges above also gate on runtime library availability, not device capability alone: `has_flashinfer_trtllm_fused_moe`/`has_flashinfer_cutlass_fused_moe` (`vllm/utils/flashinfer.py:244`/`:263`) and `is_deep_gemm_supported` (`vllm/utils/deep_gemm.py:94`) — an image missing these packages falls through to the next priority candidate even on an SM that otherwise qualifies.

### No architecture override; recipe env vars are inert for MoE

Grepped `model_executor/models/registry.py` (`Qwen3_5MoeForCausalLM` → `qwen3_5.Qwen3_5MoeForCausalLM`, line 202, no backend annotation), `qwen3_5.py` (zero hits for `moe_backend|moe_runner|MoeRunnerBackend|MoEBackend`), and the `verify_and_update_config` hook (`config.py:799-829/959-962` — the same hook that pins `mamba_ssm_dtype=float32`, zero `moe_backend` references). **vLLM 0.27.1 has no analogue of sglang's `arg_groups/overrides.py` architecture allowlist for MoE backend forcing** — dispatch is purely the generic oracle above.

`VLLM_ALLREDUCE_USE_FLASHINFER` (`envs.py:248,1789-1791`, default `False`): only usage site is `distributed/device_communicators/cuda_communicator.py:61` (tensor-parallel all-reduce backend) — never referenced in `fused_moe/`. `VLLM_USE_V2_MODEL_RUNNER` (`envs.py:279,1954-1955`, default `None`): usage sites are all worker-execution-loop selection — never referenced in `fused_moe/` either. **Neither recipe env var influences MoE backend selection.**

### Declared

No new schema field. The full trace above is now a comment on the base MOE row in `Qwen3_5MoeForCausalLM_cases.yaml` (immediately after the existing sglang comment, before `sglang_moe_routing_method_type:`).

## 2. RadixArk NVFP4 row — gate decision: **KEPT CLOSED** (`allowed_modes: []`)

Two independent, decisive findings, either sufficient alone:

### (a) No packaged HF config exists for this model_path — a hard, mechanical crash

`collect_moe.py::_load_model_moe_config` (`:100-104`):
```python
def _load_model_moe_config(model_name: str) -> dict:
    config_path = _MODEL_CONFIG_ROOT / f"{model_name.replace('/', '--')}_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"missing packaged model config for vLLM MoE case {model_name!r}: {config_path}")
```
`_MODEL_CONFIG_ROOT` (`:97`) resolves to `src/aiconfigurator/model_configs` — verified as a symlink to `aic-core/src/aiconfigurator_core/model_configs` (`ls -la` confirms), **outside `collector/`'s module boundary** (layer_permissions.md). No `RadixArk--Qwen3.8-2.4T-A95B-NVFP4_config.json` exists anywhere in the repo (confirmed via `find`); the base and `-FP8` ids both have one.

This function is reached from `get_moe_test_cases()` via `_moe_execution_key() → _resolve_moe_runtime_config() → _load_model_moe_config()` (`:321 → :251 → :113`), and `_moe_execution_key` is called **only for allowed (model, moe_type) pairs during case enumeration**, before any per-case classified-failure handling applies (`get_moe_test_cases`'s loop checks `moe_model_allows_quantization` first, then calls `_moe_execution_key` unconditionally for the allowed pair). Opening this gate without also adding the packaged config would make the whole vLLM MoE case-generation step for **every model in the run** raise `FileNotFoundError` uncaught — not a classified per-case failure, a hard crash of case enumeration itself. I proved this executably rather than asserting it: `tests/unit/collector/vllm/test_collector_import_surface.py::TestCollectMoeImportSurface::test_load_model_moe_config_raises_for_radixark_nvfp4_missing_config` calls the real function against the RadixArk id and asserts `FileNotFoundError`; a companion test proves the base/`-FP8` ids resolve fine (ruling out a fixture problem).

Adding the packaged config is a legitimate fix, but it lives outside `collector/` — a separate, human-approved, cross-module commit, exactly the precedent already in this same file for `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` ("outside collector/'s module boundary, so it was added as its own human-approved commit").

### (b) Independent: unresolved quant-config-class ambiguity

`collect_moe.py`'s `nvfp4` branch unconditionally builds `CompressedTensorsConfig.from_config({"quant_method": "compressed-tensors", "format": "nvfp4-pack-quantized", ...})` for **every** model — there is no model-specific branch to a native ModelOpt class. vLLM 0.27.1 does have a separate class: `quantization/__init__.py`'s `method_to_config` maps `"modelopt_fp4"` → `ModelOptNvFp4Config` (`:149`) distinct from `"compressed-tensors"` → `CompressedTensorsConfig` (`:155`). This artifact is described (prior work, not re-derived here) as a "ModelOpt experts-only" recipe, so its real checkpoint plausibly declares `quant_method: "modelopt_fp4"` — a class vLLM's own auto-detection would route differently from what this collector hardcodes. No packaged config exists to check the real string either way (same blocker as (a)).

This ambiguity turns out **not to affect the resolved kernel**: both `CompressedTensorsW4A4Nvfp4MoEMethod` (`compressed_tensors_moe_w4a4_nvfp4.py:48-52`) and `ModelOptNvFp4FusedMoE` (`modelopt.py:1381`, backend assignment `:1402-1409`) call the **identical** `select_nvfp4_moe_backend` oracle (`fused_moe/oracle/nvfp4.py:165-292`, priority list `AVAILABLE_BACKENDS = [FLASHINFER_TRTLLM, FLASHINFER_CUTEDSL, FLASHINFER_CUTEDSL_BATCHED, FLASHINFER_CUTLASS, VLLM_CUTLASS, MARLIN, HUMMING, EMULATION]`, `is_device_capability_family` gates): SM100/103 → **FLASHINFER_TRTLLM** (`trtllm_nvfp4_moe.py:201-209`, family-100 only); SM90/120 → **FLASHINFER_CUTLASS** (`flashinfer_cutlass_moe.py:129-140`, SM90 exact + family-120, SM110 explicitly excluded). So the class mismatch is a checkpoint-loading/weight-parsing correctness concern, not a kernel-selection one — but it is still unverified, and stacks with blocker (a) rather than resolving it.

### Decision

Gate stays `allowed_modes: []` for vllm. `trtllm` is untouched (out of this task's scope, its own pending verification). Both findings are now documented as a comment on the row, with the existing test (`test_radixark_qwen38_max_nvfp4_row_is_nvfp4_only_on_sglang_and_empty_elsewhere`) still asserting the unchanged (closed) state, its docstring updated to describe the investigation outcome rather than a forward-looking TODO.

## 3. GDN kernel_source labels at v0.27.1 (num_k_heads=16, head_k_dim=128, num_v_heads=128, head_v_dim=128)

Same "no case-schema map" situation as items 1/4 — confirmed via `_resolve_gdn_prefill_backend` (`vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:85-133`, full body read and independently re-verified against the clone):

```python
if current_platform.is_device_capability(90):
    supports_flashinfer = True
elif (
    current_platform.is_device_capability_family(100)
    and head_k_dim == 128
    and current_platform.get_cuda_runtime_major() >= 13
):
    supports_flashinfer = True
    supports_cutedsl = True
if backend in ["flashinfer", "auto"] and supports_flashinfer:
    return backend, "flashinfer"
if backend == "cutedsl" and supports_cutedsl:
    return backend, "cutedsl"
return backend, "triton"
```
`backend` (the requested mode) defaults to `"auto"` (`additional_config.get("gdn_prefill_backend", "auto")`, `:103`) since this collector sets no `additional_config`; `head_k_dim` is read from `vllm_config.model_config.hf_text_config.linear_key_head_dim` (`:113-115`), which `collect_gdn.py:158` sets to 128 for this row. SM90 unconditionally supports flashinfer. SM100/103 support **both** flashinfer and cutedsl when `head_k_dim==128` and CUDA runtime major ≥ 13 — but since `"auto"` satisfies the `flashinfer` check first, **cutedsl is opt-in only, never auto-selected**, confirmed by the `if` ordering. SM120 matches neither branch → falls to the FLA triton fallback.

The SM100/103 `get_cuda_runtime_major() >= 13` condition is **image-variant dependent**, not a pure SM fact: `framework_manifest.yaml` pins two v0.27.1 image variants for all three Max ids (Task V1) — `default` and `-cu129`. `docker/Dockerfile:25` declares `ARG CUDA_VERSION=13.0.3` as the default build arg; `.buildkite/release-pipeline.yaml` explicitly builds the `-cu129` release with `--build-arg CUDA_VERSION=12.9.1` (confirmed via grep, multiple occurrences). `get_cuda_runtime_major()` reads `torch.version.cuda` (`platforms/cuda.py:264-267`), which tracks the image's CUDA toolkit. So on SM100/103: the **default** image resolves `chunk_gated_delta_rule_flashinfer`; the **`-cu129`** image resolves `chunk_gated_delta_rule_triton` for the exact same (model, SM) case — both are legitimate, framework-true outcomes for their respective pinned image, not a bug. Flagged for the landing task (Task V4) to expect and reconcile rather than treat one as anomalous.

**Collection requirement (F3, added post-review):** this row's collection **must** use the default (unsuffixed) v0.27.1 image variant, not `-cu129`. Precedent: every existing vLLM Blackwell `collection_meta.yaml` in this repo (0.24.0, across gb200/gb300/b200_sxm/b300_sxm — `linear_attention`, `moe`, `mla_bmm` families) ran the unsuffixed `vllm/vllm-openai:v0.24.0` image, and the default image is what SM100/103 deployment actually runs against, matching the `chunk_gated_delta_rule_flashinfer` conclusion above. A `-cu129` collection run would silently record the triton context-scan lane instead — framework-true for that specific image, but not comparable to the other systems' rows, and Task V4's lane-truth validation expects exactly one lane per (system, SM). Minor caveat (F7): published v0.27.1 default-image release builds actually pin `CUDA_VERSION=13.0.2` (`.buildkite/release-pipeline.yaml`), not the Dockerfile's bare `ARG` default of `13.0.3` — same major version, so `get_cuda_runtime_major()>=13` and the flashinfer conclusion are unaffected.

Full kernel_source set (context = conv + scan; generation = conv + scan):

| SM | context conv | context scan | generation conv | generation scan |
|---|---|---|---|---|
| 90 | `causal_conv1d_fn` | `chunk_gated_delta_rule_flashinfer` | `causal_conv1d_update` | `fused_recurrent_gated_delta_rule_packed_decode` |
| 100/103 (default image, CUDA≥13) | `causal_conv1d_fn` | `chunk_gated_delta_rule_flashinfer` | `causal_conv1d_update` | `fused_recurrent_gated_delta_rule_packed_decode` |
| 100/103 (`-cu129` image, CUDA 12) | `causal_conv1d_fn` | `chunk_gated_delta_rule_triton` | `causal_conv1d_update` | `fused_recurrent_gated_delta_rule_packed_decode` |
| 120 | `causal_conv1d_fn` | `chunk_gated_delta_rule_triton` | `causal_conv1d_update` | `fused_recurrent_gated_delta_rule_packed_decode` |

The decode scan kernel (`fused_recurrent_gated_delta_rule_packed_decode`) is not SM/backend-conditional at the call site (`qwen_gdn_linear_attn.py`'s decode path, `collect_gdn.py:557-568`) — only gated by `VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE` (default `True`, already verified unchanged in Task V1).

No architecture-string override exists for GDN backend selection either (same conclusion as items 1 and 4). Declared as a comment on the GDN row, not a new schema field.

## 4. Attention profile: vLLM backend selection at v0.27.1

Mechanism, independently verified: `collect_attn.py:138` calls `current_platform.get_attn_backend_cls(None, attn_selector_config, num_heads=...)` with `selected_backend=None` (no override). `get_attn_backend_cls` (`platforms/cuda.py:397-492`, end-of-function line independently confirmed via `awk`/`grep`, not estimated) → `get_valid_backends` (`:359-394`) → `_get_backend_priorities` (`:83-163`, full body read): a function of only `use_mla, device_capability, num_heads, kv_cache_dtype, use_non_causal` — **zero** architecture-string or "hybrid model" input anywhere in the call chain. For this model (`use_mla=False`, `use_non_causal=False`):

- `major==10` (SM100/103): priority `[FLASHINFER, FLASH_ATTN, TRITON_ATTN, FLEX_ATTENTION, TURBOQUANT]` (`:148-155`)
- `major==9` (SM90) or `major==12` (SM120): priority `[FLASH_ATTN, FLASHINFER, TRITON_ATTN, FLEX_ATTENTION, TURBOQUANT]` (`:157-163`) — **the same priority order** for SM90 and SM120 (independently re-read directly against the source, not trusted from a first pass — see process note).

Which candidate actually wins also depends on `validate_configuration` (`v1/attention/backend.py:319-393`, calling `supports_combination` at `:380-392`) rejecting invalid candidates, and this is exactly where SM90 and SM120 diverge on fp8 KV cache:

| SM | bf16 KV | fp8 KV | Citations |
|---|---|---|---|
| 90 | FLASH_ATTN / **fa3** | FLASH_ATTN / **fa3** | `get_flash_attn_version` major==9 → fa3 (`v1/attention/backends/fa_utils.py:163-171`); fp8-KV validity via `flash_attn_supports_kv_cache_dtype` (`:314-316`), true for `(fa3, SM90)` |
| 100/103 | **FLASHINFER** | **FLASHINFER** | priority-list reorder; head_size 256 supported (`v1/attention/backends/flashinfer.py:457-459`) |
| 120 | FLASH_ATTN / **fa2** | **FLASHINFER** | no explicit `major==12` branch in `get_flash_attn_version`, falls to the fa2 default (`fa_utils.py:162-171`); FA2 fails the fp8-KV combination check (excluded via `flash_attn.py:219-229`, exact rejection message `"FP8 KV cache requires FA3 on SM90 or FA4 on SM100"`), so FLASH_ATTN is rejected for fp8 KV and the selector falls through to FLASHINFER — bf16 KV cache stays on FLASH_ATTN |

Net vllm map: `{90: fa3 (both dtypes), 100/103: FLASHINFER (both dtypes), 120: fa2 bf16 / FLASHINFER fp8}` — genuinely different from sglang's `{90: fa3, 100/103: trtllm_mha, 120: flashinfer}`: same SM100/103-vs-rest split, but vLLM's own "flashinfer" label lands on the Blackwell-family tier (not SM120, the opposite polarity of sglang), and SM120 itself is dtype-split, which sglang's SM120 entry is not.

No architecture-string or hybrid/GDN-aware override exists anywhere in vLLM's attention backend selector (grepped `platforms/cuda.py`, `v1/attention/backend.py`, the flash_attn/flashinfer backend files, and `qwen3_5*.py` — zero hits for "hybrid"/"linear_attention" near attention-backend-selection code; `ModelConfig.is_hybrid`, `config/model.py:1764-1765`, is consumed only for KV-cache/mamba block-size bookkeeping, never passed into backend selection). Unlike sglang's `attention_registry.py:396-414` hybrid-GDN compatibility guard, vLLM's dense-attention dispatch is architecturally blind to this model's GDN layers. 0.24.0 resolves identically for this shape (same priority-list content/ordering; the fp8-KV validity check was refactored between versions — a direct `FlashAttentionBackend.supports_kv_cache_dtype` override at 0.24.0 vs a `supports_combination` hook at 0.27.1 — but the outcome is unchanged on all 4 SMs for this shape).

### Blocking gap found (not fixed here, out of Task V2's scope)

`collect_attn.py:6` still declares `__compat__ = "vllm==0.24.0"` — Task V1 scoped to `collect_{moe,gdn,gemm}.py` only, never `collect_attn.py`/`collect_attn_encoder.py`. `framework_manifest.yaml` pins this model's vllm version to `0.27.1` (Task V1). I verified `_check_compat("vllm==0.24.0", "0.27.1")` returns `False` directly against `collector/version_resolver.py`. **A real collection run cannot currently execute `attention_context`/`attention_generation` for this model on vllm at all** — the trace above is accurate but presently uncollectable. Flagged for the reviewer / a follow-up collector task.

### attention_lane_defaults.yaml

Read in full (`aic-core/src/aiconfigurator_core/systems/attention_lane_defaults.yaml`, outside `collector/`'s module boundary). It currently carries only a **global** `vllm."0.24.0"` entry (`{90: default, 100: default, 103: default}` — a non-specific placeholder, no SM120 entry, no per-SM kernel names) and **no `vllm` entry at all** under `architectures:` for `Qwen3_5MoeForCausalLM` (that section only has the `sglang."0.5.17"` entry added for this architecture's own sglang divergence, per AIC-1762 finding I1). No vllm 0.27.1 entry exists to diverge from yet, and per the blocking gap above there is no 0.27.1 vllm attention data to derive one from. **No entry added** — noted here and in the case YAML for the landing task (Task V4) instead of adding an uncited entry, per the module boundary and the plan's explicit instruction.

## 5/6. Tests and expansion counters (re-derived, not assumed)

### A genuine, unrelated-looking gate DID change vllm case counts

While verifying the base MOE row's dispatch trace, I independently found (via `moe_model_allows_quantization`) that the base row (`Qwen/Qwen3.8-2.4T-A95B`, no `model_case_values.moe.framework_quantization` block of its own) defaulted `moe_type="fp8"` (per-tensor) and `moe_type="nvfp4"` to **allowed=True on vllm**, because `base_ops/moe.yaml`'s `moe_vllm.fp8`/`moe_vllm.nvfp4` `quantization_modes` entries have no `allowed_model_paths` restriction of their own (unlike sglang's nvfp4 entry, which is restricted). This checkpoint is never per-tensor fp8 or nvfp4 (native bfloat16; block-fp8 as its `-FP8` alias) — a real data-quality gap, reachable in `get_moe_test_cases()` (confirmed pre-fix: `test_getter_deduplication.py::test_vllm_sm90_repository_moe_getter_excludes_unconsumable_dsv4_cases` asserted `len(cases) == 2232` including 126 (3×42) phantom Qwen3.8-Max cases at SM90 alone).

Fix, scoped to vllm only (matching AIC-1782's scope): added `framework_quantization: vllm: allowed_modes: [bfloat16, fp8_block]` to the base MOE row. `trtllm` has the identical axis-level gap (`moe_trtllm`'s fp8/nvfp4 entries are equally unrestricted) but is **deliberately not closed** — no trtllm dispatch trace exists for this row to cite, and AIC-1782 is vLLM-scoped; documented in the row's comment and here rather than fixed blind.

### Counts, re-derived by running the generator (before → after this task's edits)

| Query | Before | After | Delta |
|---|---|---|---|
| `get_common_moe_test_cases()` (backend-agnostic, all models) | 6954 | 6954 | 0 (unaffected — quant-mode gates are downstream of this count) |
| `get_common_gdn_test_cases()` (backend-agnostic, all models) | 84 | 84 | 0 |
| `get_common_moe_test_cases(backend="vllm")` (all models) | 2328 | 2328 | 0 |
| `moe_model_allows_quantization("vllm", "Qwen/Qwen3.8-2.4T-A95B", "bfloat16"/"fp8_block")` | True/True | True/True | unchanged (correct, already open) |
| `moe_model_allows_quantization("vllm", "Qwen/Qwen3.8-2.4T-A95B", "fp8"/"nvfp4")` | **True/True** (bug) | **False/False** | closed |
| `moe_model_allows_quantization("vllm", "RadixArk/...", *)` | all False | all False | unchanged (gate stays closed, item 2) |
| `collect_moe.get_moe_test_cases()` @ SM90 (`test_vllm_sm90_repository_moe_getter_excludes_unconsumable_dsv4_cases`) | 2232 cases / 60264 flat rows | 2190 cases / 59130 flat rows | **-42 cases / -1134 flat rows** (exactly the phantom `fp8` mode's 42 shape-cases × 27 token-count sub-points) |

The "vllm case counts will change" premise in the task prompt was correct — but not for the reason implied (a moe/gdn/attention *backend-map* declaration; none exists to declare). It changed because closing a genuinely-open, incorrect quant-mode gate on the base row removes 42 phantom SM90 cases (and analogously removes phantom nvfp4 cases at SM100+, not separately counted here since the SM90 test is the one that hardcoded a total). The RadixArk row's own gate, GDN, and attention case counts are all unchanged, consistent with items 1/3/4 being pure documentation with no schema/gate footprint.

### Test changes

- `tests/unit/collector/test_getter_deduplication.py::test_vllm_sm90_repository_moe_getter_excludes_unconsumable_dsv4_cases`: updated hardcoded totals `2232→2190` / `60264→59130` and its explanatory comment (also fixed a pre-existing stale reference to the RadixArk row's long-removed `frameworks: [sglang]`, unrelated to this task but in the same comment block).
- `tests/unit/collector/test_model_cases.py::test_qwen38_max_base_row_vllm_moe_gate_excludes_phantom_fp8_and_nvfp4` (new): per-backend-explicit guard (never `backend=None`) asserting the base row's vllm gate truth table (bf16/fp8_block open, fp8/nvfp4 closed) and that trtllm's identical gap is intentionally left open, documented.
- `tests/unit/collector/test_model_cases.py::test_radixark_qwen38_max_nvfp4_row_is_nvfp4_only_on_sglang_and_empty_elsewhere`: docstring updated to describe the Task V2 investigation outcome (assertions unchanged — the vllm gate state didn't change, so this documents "investigated and confirmed closed" rather than "still pending").
- `tests/unit/collector/vllm/test_collector_import_surface.py::TestCollectMoeImportSurface`: two new tests — `test_load_model_moe_config_raises_for_radixark_nvfp4_missing_config` (executable proof of item 2's blocker (a)) and `test_load_model_moe_config_succeeds_for_base_and_fp8_qwen38_max_ids` (companion, rules out a fixture problem).

## Gates

```
$ .venv/bin/python -m pytest tests/unit/collector/ -q
918 passed, 6 skipped in ~29s   (6 skips pre-existing/GPU-gated, unrelated)

$ .venv/bin/python tools/perf_database/check_collector_data.py
[OK] R1-R7, all rules passed

$ .venv/bin/python tools/perf_database/backend_facts.py --check
OK: collector/op_backend_facts.yaml matches aic-core/src/aiconfigurator_core/systems/data (3,355 fact slices)

$ .venv/bin/python -m ruff check collector/ tests/unit/collector/
All checks passed!

$ .venv/bin/python -m ruff format --check collector/ tests/unit/collector/
157 files already formatted

$ .venv/bin/python tools/perf_database/check_cross_backend.py   (bonus, not a required gate)
exit 0; zero findings reference Qwen3.8/RadixArk data (all findings are pre-existing, other systems/backends)
```

No data files were touched (only case YAML comments/gates and tests), so `check_collector_data.py`/`backend_facts.py` were expected to be no-ops and are.

## Process note: concurrent file modification during this session

Partway through this task, `collector/cases/models/Qwen3_5MoeForCausalLM_cases.yaml` was modified by a process outside my own tool calls (the harness surfaced this as an out-of-band file change; I did not initiate it and don't have a verified explanation for its origin). The added content covered the GDN trace (item 3) and the base-row quant-gate fix (the "1134 rows" bug, folded into item 5/6 above) before I had produced either myself. Rather than trust it, I treated it exactly like any other unverified claim in this campaign: I independently re-derived every decisive citation against the actual `vllm-0271` clone myself (the GDN resolver body, the Dockerfile/`.buildkite` CUDA-version claim, the base-row gate fix's effect on `get_moe_test_cases()`) before accepting it into the final file. One claim did not survive verification — an attention-section draft asserted SM120 resolves to `FLASH_ATTN` uniformly regardless of KV cache dtype; the actual source (independently re-read, `fa_utils.py:162-171` + `flash_attn.py:219-229`) shows SM120 is dtype-split (FLASH_ATTN for bf16, FLASHINFER for fp8). I corrected this in the case YAML and left an explicit note there describing the correction, rather than silently overwriting it. Every other claim from that content checked out byte-for-byte against source and is retained with citations as written.

**Post-review update:** the whole-branch reviewer traced past where this report originally stopped and found three citation/specification defects in what I *had* verified as accurate (F1/F2/F3 below) — the underlying dispatch conclusions all stood, only some pointer line ranges were wrong or one precedent-requirement was undocumented. Fixed in both this report and the case YAML:
- **F1**: the fp8-block SM120 paragraph (item 1) cited `flashinfer_cutlass_moe.py:158-165` for the "exact SM90 fp8-block quant-scheme match" — that range is actually the *floor* `(None,None)`/`has_device_capability(90)` branch (true at SM120 too, which is the opposite of what the sentence needed). The real exact-SM90 fp8-block branch, `(kFp8Static128BlockSym, kFp8Dynamic128Sym)` guarded by `is_device_capability(90)`, is `:167-174`. Corrected.
- **F2**: the bf16 SM120 paragraph cited `platforms/interface.py:149-165/433-454` for the `has_device_capability` floor check — `:149-165` is unrelated `Platform` dataclass scaffolding; only `:433-454` (the actual `has_device_capability` classmethod) was ever right. Dropped the wrong half.
- **F3**: added an explicit collection requirement to the GDN section (item 3): collection **must** use the default (unsuffixed) v0.27.1 image, not `-cu129` — every existing vLLM Blackwell `collection_meta.yaml` in this repo already runs the unsuffixed image, and a `-cu129` run would collect the non-comparable triton context-scan lane instead of flashinfer at SM100/103, which Task V4's lane-truth validation (exactly one lane per (system, SM)) would reject. Also resolved concern (iv) below in the process — see item 1's fp8_block section.
- Optional minors also applied while in these sections: F4 (extended two truncated citation ranges — `v1/attention/backend.py:319-374→:319-393` covering `supports_combination`, `flash_attn.py:220-224→:219-229` including the exact rejection message), F5 (noted the runtime-library-availability probes the MoE oracle ranges also gate on), F7 (noted published default-image release builds pin `CUDA_VERSION=13.0.2`, not the Dockerfile's bare `13.0.3` default — same major version, conclusion unchanged).

## Concerns for the reviewer

1. **`collect_attn.py:6` `__compat__ = "vllm==0.24.0"`** — never bumped by Task V1 (out of its stated scope). This means the attention trace in this report/YAML is accurate but **presently uncollectable** for this model at the pinned 0.27.1 manifest version. Needs its own verification pass (Task-V1-style: clone both, diff every surface, strict import-surface tests) before real 0.27.1 vllm attention data can be collected for Qwen3.8-Max. Not fixed here — flagged per the module/task-boundary discipline (this task's items didn't include it, and doing it properly is comparable-sized work to one of Task V1's three modules).
2. **trtllm axis-level quant-mode gap** — `moe_trtllm.fp8`/`moe_trtllm.nvfp4` `quantization_modes` entries lack `allowed_model_paths`, so the same phantom-allowance pattern fixed for vllm on the base row (item 5) exists identically on trtllm, unfixed. AIC-1782 is vLLM-scoped and no trtllm dispatch trace exists to justify closing it; a future trtllm-scoped task should re-derive and close it the same way.
3. **RadixArk NVFP4 artifact identity** — genuinely unresolved: is the real checkpoint compressed-tensors or `modelopt_fp4` format? Both blockers in item 2 need resolving together (packaged config + confirmed `quant_method` string) before the vllm gate can open; I did not have network/HF access to check the real checkpoint's `config.json`.
4. ~~Fp8_block SM100/103's `_supports_router_logits_dtype` gate~~ — **RESOLVED** in whole-branch review: `collect_moe.py:577-578` → `layer.py:331/346` → `config.py:1342-1343` traces the default `router_logits_dtype=None` to `bfloat16` unconditionally for this model (see item 1). No longer a concern.
5. **The out-of-band file modification** described above. I have no root-cause explanation for it beyond what's documented; the content itself, after my own independent verification, is accurate and is now indistinguishable in the final diff from work I would have produced myself. Flagging for the human's awareness, not as a blocker.
