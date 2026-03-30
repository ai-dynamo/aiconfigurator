# Guard Rails Reference

Distilled from ~60 merged PRs. These are constraints that MUST NOT be violated.
Each guard has a severity level indicating the impact of violation.

Severity: **CRASH** (engine won't start), **OOM** (out of memory), **SILENT** (wrong results),
**K8S** (pod failure), **PERF** (degraded performance)

---

## TRT-LLM Backend

| Guard | Severity | Key Detail |
|---|---|---|
| `build_config` nesting is version-dependent | SILENT | Pre-1.2.0rc5: inside `build_config`. Post-1.2.0rc5: top-level. Wrong placement silently ignored. |
| Template variables must be top-level keys | SILENT | `engine.py` flattens context; nested paths like `build_config.max_batch_size` are undefined. |
| `cache_transceiver_config` lifecycle | CRASH | Absent pre-1.0.0rc4. In engine YAML 1.0.0rc4-1.3.0rc1. In CLI `--override-engine-args` post-1.3.0rc5. |
| `cache_transceiver_config.backend` must be `'DEFAULT'` | CRASH | Omitting backend field causes TRT-LLM to reject config in disagg mode. |
| Engine params go into `--override-engine-args` JSON | CRASH | TRT-LLM's Dynamo argparser only accepts specific direct CLI flags. |
| `max_num_tokens % tokens_per_block == 0` | CRASH | TRT-LLM asserts block alignment; violation crashes engine startup. |
| `cache_transceiver_max_tokens_in_buffer` must align to block | CRASH | Same block-alignment assertion applies. |
| Prefill: `disable_overlap_scheduler = true` | CRASH | Overlap scheduler on prefill causes hangs. |
| Decode/agg: `disable_overlap_scheduler = false` | PERF | Disabling on decode/agg hurts throughput. |
| MoE: `TP = moe_tp * moe_ep` | OOM | Without this, model may replicate per GPU. |
| KV cache dtype `"float16"`/`"bfloat16"` -> `"auto"` | CRASH | TRT-LLM doesn't accept literal dtype strings. |

## SGLang Backend

| Guard | Severity | Key Detail |
|---|---|---|
| MoE: `TP = moe_tp * moe_ep` | OOM | SGLang uses unified TP; without this, `tp=1` -> full model replicated per GPU. |
| Do NOT emit `--moe-dense-tp-size` | CRASH | SGLang only accepts value 1 or None; other values crash startup. |
| KV transfer backend default: `"nixl"` | CRASH | SGLang >=0.5.6.post2 requires explicit transfer backend for disagg. |
| `enable_mixed_chunk = true` for agg mode | PERF | Without it, poor prefill/decode scheduling. |
| KV cache dtype `"fp8"` -> `"fp8_e4m3"` | SILENT | SGLang accepts `"fp8_e4m3"` not `"fp8"`. |
| KV cache dtype `"float16"` -> `"auto"` | SILENT | SGLang interprets `"float16"` literally. |
| Wideep vs non-wideep are exclusive | CRASH | SGLang doesn't support mixed `moe_tp + moe_ep` configurations. |
| `enable_attention_dp` when DP > 1 and MoE | SILENT | MoE models with DP>1 require attention DP. |
| NVFP4 models need explicit `--quantization` | OOM | SGLang can't auto-detect NVFP4; loads in FP16/BF16 without it. |

## vLLM Backend

| Guard | Severity | Key Detail |
|---|---|---|
| `--cudagraph-capture-sizes` uses space-separated values | CRASH | Commas treated as part of value strings; startup failure. |
| `--kv-transfer-config` required for disagg | SILENT | Without it, KV cache silently fails to transfer. |
| `--max-model-len` and `--max-num-batched-tokens` required | OOM | Without these, vLLM uses model default max length. |
| `enable_expert_parallel` when `moe_ep > 1` | SILENT | vLLM requires explicit flag; EP silently disabled without it. |
| KV cache dtype `"float16"` -> `"auto"` | CRASH | vLLM doesn't accept `"float16"` as literal. |

## Cross-Backend Rules

| Guard | Severity | Key Detail |
|---|---|---|
| `gpus_per_worker = TP * PP * DP` | K8S | Wrong value causes pod scheduling failure or idle GPUs. |
| Prefix caching: `disable_prefix_cache=false` AND `enable_router=true` | SILENT | Both must be set when `ModelConfig.prefix > 0`. |
| Speculative decoding type string differs by backend | CRASH | TRT-LLM: `"MTP"`, SGLang: `"NEXTN"`, vLLM: `"mtp"`. |
| Prefill `max_batch_size` must never be 0 | CRASH | Fallback to 1; zero causes division-by-zero. |
| Do NOT extract common K8s templates | N/A | PR #314 attempted; PR #340 reverted. Keep templates standalone per backend. |

## K8s Template Guards

| Guard | Severity | Key Detail |
|---|---|---|
| PVC mounts on WORKER pods, not frontend | K8S | SGLang frontend doesn't need model files. |
| `k8s_model_cache` must be present for vllm/sglang | K8S | Missing PVC param omits volume mount entirely. |
| `k8s_etcd_endpoints` must be rendered when configured | K8S | Disagg mode requires etcd for service discovery. |
| `k8s_hf_home` defaults to `/workspace/model_cache` when PVC set | SILENT | Ensures HF downloads go to persistent volume. |
| DGD name must be RFC 1123 compliant | K8S | Lowercase, alphanumeric, hyphens only, max 63 chars. |
| Use `MODEL_PATH` not `MODEL` as env var | SILENT | Aligns with trtllm convention; `MODEL` caused conflicts. |

## Version & Template Selection

| Guard | Severity | Key Detail |
|---|---|---|
| Template version selection: floor match (highest <= requested) | SILENT | Not every version has its own template. |
| Catch unsupported system/backend/version combos early | PERF | Late failures after profiling waste user time. |
| `--generator-dynamo-version` vs `--generated-config-version` | SILENT | Different purposes: Dynamo image tag vs template selection. |

## Run Script Guards

| Guard | Severity | Key Detail |
|---|---|---|
| `#!/bin/bash` shebang and `set -e` | CRASH | Missing shebang causes exec format error. |
| Benchmark templates must include `--artifact-dir` | CRASH | Default path may be read-only in containers. |
| Dynamo >=0.8.0 entry points: `python3 -m dynamo.<backend>` | CRASH | Old entry points fail on new Dynamo versions. |
| Multi-node disagg: per-node scripts, `include_frontend` only node 0 | K8S | Workers assigned to non-existent GPU slots. |

## Model Detection

| Guard | Severity | Key Detail |
|---|---|---|
| MoE detection from `config.json`, not model name | SILENT | Name patterns miss non-standard naming. |
| Use `model_family` not `model_name` for compatibility | SILENT | `model_name` includes size/quant info. |
| `safe_model_name` generated BEFORE saving results | CRASH | Race condition with raw `model_path` containing slashes. |
| Quantization inferred from model config, not name | SILENT | Name patterns are inconsistent. |

## Benchmark vs Deployment Differences

| Parameter | Deployment | Benchmark |
|---|---|---|
| `agg_decode max_batch_size` | `max(512, batch*2)` | `max(128, batch)` |
| `cuda_graph_batch_sizes` | Coarse-grained ranges | All sizes `range(1, max+1)` |
| Rule selection | `--generator-set rule=default` | `--generator-set rule=benchmark` |
