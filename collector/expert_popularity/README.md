# Expert popularity collection

This standalone whole-model operation records the logical routed-expert choices made by
SGLang serving. It produces model-specific data, not GPU performance data. Raw recorder,
request, response, server, and Slurm evidence stays in the campaign directory; only a
validated two-file bundle is packaged.

## Canonical workload

- SGLang 0.5.14 using either native `stat` recorder mode or a fail-closed
  serving-selected routed-expert capture documented in bundle provenance.
- Prefill only; sequential requests; one ignored generated token; temperature zero.
- Four fixed seed shards, each with at least 65,536 prompt tokens.
- ISL sampled from the discrete uniform distribution over `[128, 4096]`.
- Input IDs sampled uniformly from tokenizer vocabulary IDs after excluding special IDs.
- The entire workload is run twice. Counts must be exactly repeatable by default;
  models whose serving kernel cannot replay routes exactly may use the explicit
  aggregate gate documented below, while retaining every raw per-request route.
- Each routed layer must conserve `routed_token_count * top_k` assignments.

The packaged Parquet file has one row per logical `(layer_id, expert_id)` and the columns
`layer_id`, `expert_id`, `activation_count`, `routed_token_count`, `token_hit_rate`,
`assignment_share`, and `popularity_rank`. Hardware and serving details are provenance in
the private campaign artifacts; they are not lookup keys and are never packaged. Public
`metadata.yaml` retains only model/checkpoint identity, framework/image identity, observer
method, validation evidence, and content digests.

## Generic Slurm launchers

The launchers under `slurm/` contain no cluster account, partition, host, network-device,
or user-path defaults. The submitter must provide scheduler resources and the following
site-local paths through `sbatch` options and exported environment variables:

- `CAMPAIGN_ROOT`: private raw-artifact directory
- `IMAGE_SQSH`: local immutable container image
- `IMAGE_REFERENCE`: immutable public image identity
- `HF_CACHE`: model cache
- `GPUS_PER_NODE`: allocated GPUs per node for `multinode.sbatch`

Partition, account, node count, GPU count, constraints, topology preferences, and log
destination belong in the site-local submit command or wrapper, not in this repository.
NCCL/UCX interface selection is inherited from the submitting environment. The checked-in
scripts are `single_gpu.sbatch`, `multinode.sbatch`, `stage_model.sbatch`, and
`stage_sglang_image.sbatch`.

## Collection registry

This is the complete model-specific expert-popularity target list. `COLLECTING`
means an immutable-revision stage/smoke/production dependency chain is queued
or running, but no production result exists yet. `PLANNED` means that the model
is in scope but has not been submitted. Neither status authorizes a floating
checkpoint. Every run resolves an immutable revision before staging. Quantized
checkpoints are validation candidates, not assumed aliases of the canonical
model.

| Status | Priority | Canonical bundle identity | Architecture | Collection checkpoint candidates |
|---|---:|---|---|---|
| `PUBLISHED` | P0 | `deepseek-ai/DeepSeek-R1` | `DeepseekV3ForCausalLM` | revision `56d4cbbb4d29f4355bab4b9a39ccb717a14ad5ad` |
| `PUBLISHED` | P0 | `deepseek-ai/DeepSeek-V3` | `DeepseekV3ForCausalLM` | revision `e815299b0bcbac849fa540c768ef21845365c9eb` |
| `PUBLISHED` | P0 | `deepseek-ai/DeepSeek-V3.1` | `DeepseekV3ForCausalLM` | revision `c0781d039fb7a1ba2abc4add0bdc293e92d2b8db` |
| `PUBLISHED` | P0 | `deepseek-ai/DeepSeek-V3.2` | `DeepseekV32ForCausalLM` | revision `a7e62ac04ecb2c0a54d736dc46601c5606cf10a6` |
| `PUBLISHED` | P0 | `MiniMaxAI/MiniMax-M2.7` | `MiniMaxM2ForCausalLM` | revision `d494266a4affc0d2995ba1fa35c8481cbd84294b` |
| `PUBLISHED` | P0 | `moonshotai/Kimi-K2.5` | `KimiK25ForConditionalGeneration` | revision `4d01dfe0332d63057c186e0b262165819efb6611` |
| `PUBLISHED` | P0 | `zai-org/GLM-5.2` | `GlmMoeDsaForCausalLM` | revision `b4734de4facf877f85769a911abafc5283eab3d9` |
| `PUBLISHED` | P0 | `moonshotai/Kimi-K2.7-Code` | `KimiK25ForConditionalGeneration` | revision `74797c9c62378b951a1f6fcf5c4631024e9b8bef` |
| `PUBLISHED` | P0 | `Qwen/Qwen3-235B-A22B` | `Qwen3MoeForCausalLM` | revision `8efa61729e24bd65b1d152b5ab5409052aa80e65` |
| `PUBLISHED` | P0 | `deepseek-ai/DeepSeek-V4-Flash` | `DeepseekV4ForCausalLM` | canonical revision `60d8d70770c6776ff598c94bb586a859a38244f1`; collected through routing-equivalent `sgl-project/DeepSeek-V4-Flash-FP8@ae01d80c06cdfe30581edfd0e1c5449dc7ed7f17` |
| `PUBLISHED` | P0 | `deepseek-ai/DeepSeek-V4-Pro` | `DeepseekV4ForCausalLM` | canonical revision `b5968e9190ef611bbf34a7229255be88a0e937c1`; collected through routing-equivalent `sgl-project/DeepSeek-V4-Pro-FP8@54eeff4ae56c7605c99bbb8b5fcd54412745fb5f` |
| `PUBLISHED` | completed slice | `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct` | `DeepseekV2ForCausalLM` | revision `e434a23f91ba5b4923cf6c9d9a238eb4a08e3a11` |
| `PUBLISHED` | completed slice | `Qwen/Qwen1.5-MoE-A2.7B` | `Qwen2MoeForCausalLM` | revision `1a758c50ecb6350748b9ce0a99d2352fd9fc11c9` |
| `PUBLISHED` | completed slice | `Qwen/Qwen3-30B-A3B` | `Qwen3MoeForCausalLM` | revision `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` |
| `PUBLISHED` | completed slice | `openai/gpt-oss-20b` | `GptOssForCausalLM` | revision `6cee5e81ee83917806bbde320786a8fb61efebee` |

On GB200, SGLang 0.5.14 auto-selects the fused `flashinfer_trtllm` backend for
Qwen3 and `flashinfer_mxfp4` for GPT-OSS. Both initial smokes returned an all-zero
recorder matrix because their internal/bypassed routing path does not invoke the
recorder's `select_experts` hook. The production-compatible remedy uses the
routed FlashInfer path: SGLang materializes and records explicit expert IDs, and
the MoE kernel consumes those same IDs. For `flashinfer_trtllm` this is the
native `flashinfer_trtllm_routed` backend. For `flashinfer_mxfp4`, the collector
uses a fail-closed, source-hash-pinned bridge only during active recorder windows.
Neither path publishes data until its conservation and repeatability gates pass.
Both routed-path production collections passed those gates on GB200 and are
published above.

Kimi K3 remains a later campaign: its completed TP16 POC required a runtime source overlay.
Production publication requires a digest-pinned image containing the integration and strict
base/checkpoint routing-equivalence evidence.

The EP support list supplies priority only. Its `PASS`, `NO_DATA`, and
`HW_INCOMPATIBLE` labels describe the separate EP performance database and do
not prove that SGLang's expert-distribution recorder observes the
serving-selected routing path. Every registry row must independently pass
observer smoke, exact assignment conservation, repeat stability, and cross-seed
stability gates. Repeat validation is exact by default. A model may use the
explicit `aggregate` mode only when the serving framework cannot provide exact
route replay; that mode records the non-exact result and requires mean-layer
Pearson at least `0.999` and mean-layer Jensen-Shannon divergence at most
`0.001`. Raw per-request routes remain part of the evidence.

Before scheduling, every candidate receives an immutable model revision and an
estimated minimum topology. Floating model names in this registry are never
used as collection identities. DeepSeek V4 Flash and Pro stay separate because
their popularity matrices are model-specific even when other performance-table
consumers canonicalize some module shapes between them.

`sgl-project/DeepSeek-V4-Pro-FP8` is collected at TP8. SGLang 0.5.14 rejects
TP16 for this block-FP8 checkpoint because the shared-expert output partition is
192 elements, which is not divisible by the 128-element quantization block.
This is a checkpoint/topology incompatibility, not a transient cluster failure;
TP16 failures must not be retried as infrastructure errors.

Long-context DeepSeek V4 requests can trigger TorchInductor compilation after
the server health check has already passed. For dependent large/production
jobs, `COMPILE_CACHE_HOST_DIR` may mount a persistent cache at
`/compile-cache`, while a model-, revision-, topology-, and GPU-architecture-
specific `COMPILE_CACHE_KEY` prevents incompatible reuse. The runner gives
each Slurm node rank a separate cache directory so independent nodes never
replace the same FlashInfer or DeepGEMM lock file. It also defaults
`TORCHINDUCTOR_COMPILE_THREADS` to 8. Cache paths, scheduler identity, hardware
UUIDs, node names, and job IDs remain private campaign diagnostics; they are
never copied into packaged model-popularity metadata.

`CONTEXT_LENGTH`, `MAX_TOTAL_TOKENS`, `MAX_PREFILL_TOKENS`, and
`CHUNKED_PREFILL_SIZE` are explicit runner inputs. A smaller chunked-prefill
size changes only the serving forward boundary: the request still contains the
full sampled ISL, and the response capturer reconstructs all request-token
routes from the request-to-token cache before validation. The exact resolved
values remain in `server_args.json`; no run may infer or truncate the requested
ISL to satisfy a backend limitation.

## Repacked checkpoint provenance

The serving checkpoint may differ from the canonical bundle identity only when a
fail-closed routing-equivalence report passes. The report verifies the immutable
IDs and revisions, routing-relevant configuration, weight-index identity,
explicitly accounted physical tensor deltas, every router tensor byte-for-byte,
exact canonical-FP8-to-checkpoint-BF16 `wo_a` dequantization, tokenizer
artifacts, required runtime semantics, and the
checkpoint's weight-only/no-retraining provenance claim. The bundle records both
`model.{id,revision}` and `provenance.collection_checkpoint`; it never presents a
repacked checkpoint as the canonical model. `verify_routing_equivalence.py`
generates the evidence, and the collection driver rejects non-canonical
checkpoints without matching passing evidence.
