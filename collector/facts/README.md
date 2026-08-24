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

Rule 6 (2026-08-23, from the DSV4 false positive): when a framework probes WEIGHT-FILE properties — not just config — the dummy must carry that signal. sglang picks DSV4 expert layout from the safetensors header dtype; weightless dummies fell to an env default and misclassified converted-FP8 checkpoints. gen_dummy_models now writes a tiny `dtype_probe.safetensors` mirroring the real checkpoint's routed-expert key+dtype.

## Components

| File | Role |
|---|---|
| `targets.yaml` | INPUT only: what to probe and how — platform, backend image/version pins, model roster (derived from collector cases + owner extras/exclusions), per-model generate args (`cli_extra_args`, each with a `fact:` citation) and dummy overrides. |
| `results/<sm>/<framework>.yaml` | THE consolidated results (`gen_facts.py --matrix`): per checkpoint — verdict (pass / pass+custom / fail), extra generate args when needed, failure cause, and the deployed identity actually measured (attention backend, MoE quant→kernel family, allocated KV dtype, topology). One file per (SM, framework, version) — the version is in the filename (`sglang-0.5.16.yaml`), so results for multiple versions coexist side by side. targets.yaml pins exactly one version per backend; bumping the pin produces a NEW result file. Future SMs are sibling dirs. |
| `findings.yaml` | OUTPUT: curated conclusions from probe campaigns, one entry per finding with `applies_to` scopes and inline evidence/versions/dates. Separate lifecycle from targets.yaml — findings rot with framework releases and are re-verified on version bumps. |
| `gen_dummy_models.py` | HF config -> depth-cut dummy model dirs. Width is NEVER shrunk (TP divisibility and quant shape checks must behave like the real checkpoint). One variant per interleaved layer kind; per-layer quant-config entries filtered and renumbered; a post-check fails loudly on any surviving out-of-range layer reference. |
| `gen_facts.py` | THE driver (single entry point): `--emit-queues` targets -> golden `cli generate` renders -> per-GPU probe queues (`--only` scopes to a repo subset); `--records` raw probe JSONs -> curated `records.jsonl` (kernel normalization + taxonomy backend labels + compressed errors); `--matrix` -> `results/<sm>/<framework>-<version>.yaml`; `--check-coverage` collector-roster lower bound. Also owns checkpoint quant-profile derivation (from quant metadata, never repo names). |
| `probe_sglang.py` | sglang in-container probe. `--engine-cli` parses generator output through sglang's own CLI parser; overrides (dummy load, KV-pool cap, cuda-graph off) are appended as CLI flags so `ServerArgs.__post_init__` sees them. `--trace` runs one eager prefill+decode under the profiler. |
| `probe_vllm.py` | vLLM probe via the FPM path: parse `run.sh`'s `engine_command`, strip FPM-owned flags, feed vLLM's `EngineArgs.from_cli_args`, in-process EngineCore, generic attention-class scan on the loaded model. |
| `probe_trtllm.py` | TRT-LLM probe: llmapi with `TLLM_WORKER_USE_SINGLE_PROCESS=1` (in-process worker), dummy load, kernel capture. Includes narrowly-scoped shims for a broken cutlass-DSL package walk (needed on both 1.3.0rc20 and rc23 images; the stub tries the real import first and only fills genuine holes, documented inline). |
| `kernel_taxonomy.yaml` | Observed CUDA kernel name -> canonical backend (regex rules, first match wins). Uses the SAME backend vocabulary as `collector/kernel_source_backends.yaml`, so probe evidence and collector claims translate through one namespace. Seeded from a 258-kernel SM90 inventory; unmatched kernels are the file's backlog signal. |
| `configs/repos.txt` | The probe roster (every checkpoint the collector's case yamls mention, plus probe-only additions). `gen_facts.py --check-coverage` enforces the collector roster as a coverage LOWER bound; owner-decided exclusions live in `targets.yaml roster.excluded` (decided_by/reason required). |

## Usage

```bash
# 1. dummy models (fetch configs once; AIC model_configs/ are reused when present)
python3 collector/facts/gen_dummy_models.py --configs <cfg_dir> --out <ws>/dummy_models

# 2. plan + queues (renders engine args from this repo's generator)
AIC_PROBE_WORKSPACE=<ws> python3 collector/facts/gen_facts.py --emit-queues --backends sglang,vllm,trtllm
bash <ws>/archive/queues/gpu0.sh   # ... one per GPU; done-guard makes reruns incremental

# 3. collect + curate
AIC_PROBE_WORKSPACE=<ws> python3 collector/facts/gen_facts.py --records
AIC_PROBE_WORKSPACE=<ws> python3 collector/facts/gen_facts.py --matrix
```

A full three-backend sweep of the 76-checkpoint roster is ~226 runs, minutes
each on one GPU per run (DeepGEMM/flashinfer JIT warmup dominates; mount a
shared cache to amortize). Every checkpoint runs individually — architecture
dedup is deliberately NOT done, because quant-artifact siblings of one
architecture have been observed to select different backends.

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
  not yet implemented); tp in {1,2,8} run real (8-GPU rank-injection probe).
- See `UPGRADE_GATE.md` for the proposed process change that consumes this
  module: facts-before-upgrade as playbook step 0, and facts-record citations
  replacing source-reading citations in the collector rules.
