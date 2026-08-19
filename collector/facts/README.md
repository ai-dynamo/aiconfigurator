# collector/facts — runtime identity probing for collector/generator correctness

Probes what a serving framework ACTUALLY does — which quant method binds to
each module, which Python API an op flows through, which CUDA kernels execute —
instead of deriving those facts by reading framework source. The output is a
machine-generated facts archive that collectors, `op_backend_facts.yaml`, the
generator, and upgrade agents can be validated against.

Motivation: every SGLang/TRT-LLM/vLLM version bump historically broke
collectors through *silently wrong* facts (default backend changed, import
moved, quant identity mis-bound) that code reading failed to catch. This
harness replaces inferential code reading with execution: dummy-weight models
(depth-cut, width-true), generator-rendered engine args as probe input (zero
translation drift), and three-level identity capture.

## The three-level facts contract

For each (checkpoint, quant profile, backend, version, topology):

- **A_quant** — per-module `quant_method` classes, weight dtypes/shapes.
  *"the quant judgment is right"*
- **B_api** — API-boundary spans (`torch.profiler.record_function` wraps on
  MoE dispatch, attention forward, quant `apply`) with Python `file.py(line)`
  call chains. *"the collector calls the same API"*
- **C_kernels** — CUDA kernels grouped under each span via the profiler event
  tree, plus unattributed "orphan" kernels as a coverage signal.
  *"the benchmark executes the same kernels"*

Rejections and crashes are structured facts, not failures (e.g. "native DSV4
W4A8 on SM90: sglang binds Fp8MoEMethod over fp4-packed weights and crashes at
first MoE forward; vLLM routes it to fused_marlin_moe and runs").

## Components

| File | Role |
|---|---|
| `targets.yaml` | Target matrix: model families x checkpoints (default / FP8 / NVFP4 / MXFP4...), KV pairing copied from AIC `_PROFILE_TO_QUANT`, backend images, dummy-variant taxonomy, per-backend overrides. Known findings are recorded inline as `known:` fields. |
| `gen_dummy_models.py` | HF config -> depth-cut dummy model dirs. Width is NEVER shrunk (TP divisibility and quant shape checks must behave like the real checkpoint). One variant per interleaved layer kind; per-layer quant-config entries filtered and renumbered; a post-check fails loudly on any surviving out-of-range layer reference. |
| `gen_facts.py` | Driver: targets -> generator-rendered engine args (`render_backend_templates`, so the probe runs exactly what a deployment would) -> per-GPU probe queues -> `archive.jsonl` with provenance (incl. generator src commit). |
| `probe_sglang.py` | sglang in-container probe. `--engine-cli` parses generator output through sglang's own CLI parser; overrides (dummy load, KV-pool cap, cuda-graph off) are appended as CLI flags so `ServerArgs.__post_init__` sees them. `--trace` runs one eager prefill+decode under the profiler. |
| `probe_vllm.py` | vLLM probe via the FPM path: parse `run.sh`'s `engine_command`, strip FPM-owned flags, feed vLLM's `EngineArgs.from_cli_args`, in-process EngineCore, generic attention-class scan on the loaded model. |
| `probe_trtllm.py` | TRT-LLM probe: llmapi with `TLLM_WORKER_USE_SINGLE_PROCESS=1` (in-process worker), dummy load, kernel capture. Includes narrowly-scoped shims for a broken cutlass-DSL package walk in the 1.3.0rc20 image (documented inline). |
| `make_records.py` | Raw facts -> curated records: kernel-name normalization, infrastructure-noise filters, nested-span merge, compressed errors. Raw JSONs remain as evidence; records are the consumption layer. |

## Usage

```bash
# 1. dummy models (fetch configs once; AIC model_configs/ are reused when present)
python3 collector/facts/gen_dummy_models.py --configs <cfg_dir> --out <ws>/dummy_models

# 2. plan + queues (renders engine args from this repo's generator)
AIC_PROBE_WORKSPACE=<ws> python3 collector/facts/gen_facts.py --emit-queues --backends sglang,vllm,trtllm
bash <ws>/archive/queues/gpu0.sh   # ... one per GPU; done-guard makes reruns incremental

# 3. collect + curate
AIC_PROBE_WORKSPACE=<ws> python3 collector/facts/gen_facts.py --collect
AIC_PROBE_WORKSPACE=<ws> python3 collector/facts/make_records.py
```

A full three-backend sweep of 12 checkpoints is ~36 runs, minutes each on one
GPU per run (DeepGEMM/flashinfer JIT warmup dominates; mount a shared cache to
amortize).

## Selected findings from the first sweeps (SM90, details in records)

- sglang silently loads MiniMax-M3-NVFP4 as Unquantized (vLLM correctly binds
  ModelOptNvFp4FusedMoE) — the exact "green run, wrong identity" failure class
  this harness exists to catch.
- NVFP4 MoE on SM90 executes via Marlin bf16 dequant on every framework that
  runs it — perf data collected there is not fp4-tensor-core data.
- GLM-5.2 DSA on SM90: auto KV resolves bf16 -> prefill flashmla_sparse /
  decode fa3, with a dense-FA3 branch below index_topk=2048; explicit fp8 KV
  (what the generator emits for fp8 profiles) -> flashmla_kv both phases, with
  per-step online K-cache quantization inside the decode path.
- Generator gaps: `tokens_per_block` is an unvalidated passthrough (64 breaks
  MiniMax-M3 AND DeepSeek-V4 on vLLM); fp8-profile GLM + vLLM + SM90 renders a
  deployment with no usable attention backend. The probe doubles as the boot
  check the generator currently lacks.

## Known limitations

- Engine args are generator-faithful for sglang/vLLM; the TRT-LLM path uses
  probe defaults (extra_engine_args YAML fidelity is a pending increment).
- Identity probing only — no performance numbers (dummy weights distort
  data-dependent paths; that is the collectors' job).
- tp/ep > 2 selection facts require capability-mocked enumeration (designed,
  not yet implemented); tp in {1,2} run real.
- Kimi-K3 dummy adapter pending (VL wrapper + hybrid linear-attention pattern).
