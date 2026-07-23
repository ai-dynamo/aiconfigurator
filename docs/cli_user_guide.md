# CLI User Guide
## Basic Command
As mentioned in root Readme, CLI supports five modes: `default`, `exp`, `generate`, `estimate`, and `support`. We'll go through these modes one by one.

Quantization defaults are inferred from the Hugging Face model config (`config.json` plus optional `hf_quant_config.json`).  
For low-precision models, use a quantized HF ID (for example, `Qwen/Qwen3-32B-FP8`) or a local model directory containing those files.

## Common Arguments (all modes)

These flags are shared across modes (a few are sweep-only, as noted):

- `--log-level LEVEL`: Set the minimum log level (`CRITICAL`, `ERROR`, `WARNING`, `INFO`, `DEBUG`). Priority: `--log-level > AICONFIGURATOR_LOG_LEVEL > --debug > INFO`. (all modes)
- `--debug`: Deprecated alias for `--log-level DEBUG`, kept for backward compatibility. Prefer `--log-level`. (all modes)
- `--no-color`: Disable ANSI colors in output. (all modes)
- `--save-dir DIR`: Directory to write results and generated deployment artifacts. (`default`, `exp`, `generate`, `estimate`)
- `--top-n N`: Number of top configurations to output — per experiment in `exp` mode, or per serving mode (agg/disagg) in `default` mode. Default: `5`. (`default`, `exp`, `generate`, `estimate`)
- `--systems-paths`: System search paths (comma-separated). Use `default` for the built-in systems path; the first match wins for an identical system/backend/version. (`default`, `exp`, `generate`, `estimate`)
- `--deployment-target`: Generated-artifact platform — `dynamo-j2` (default), `dynamo-python`, `llm-d-helm`, `llm-d-kustomize`, or `fpm`. See [Deployment Target Selection](#deployment-target-selection). (`default`, `exp`, `generate`, `estimate`)
- `--engine-step-backend`: Experimental static-latency backend — `python` (default) or `rust` (routes static step estimates through the Rust FPM estimator). (`default`, `exp`, `generate`, `estimate`)

The `support` mode accepts only `--log-level`, `--debug`, and `--no-color` from this list. Generator-artifact flags (`--generator-config`, `--generator-set`, `--generator-help`, `--generator-help-backend`, `--generated-config-version`, `--generator-dynamo-version`) are documented under [Default mode](#default-mode).

## Defaults and Implicit Behavior

When using `default` mode, several parameters have default values that affect
which configurations are considered feasible. These defaults are applied when
the corresponding flag is not specified:

| Parameter | Default | Flag | Effect |
|-----------|---------|------|--------|
| ISL (Input Sequence Length) | 4000 | `--isl` | Assumed input prompt length |
| OSL (Output Sequence Length) | 1000 | `--osl` | Assumed output generation length |
| TTFT (Time to First Token) | 2000 ms | `--ttft` | Max acceptable time to first token |
| TPOT (Time per Output Token) | 30 ms | `--tpot` | Max acceptable time per output token |
| Strict SLA | off | `--strict-sla` | Pre-filter Pareto frontier to only SLA-compliant configs |
| Inclusive TPOT | off | `--inclusive-tpot` | Report TPOT inclusive of TTFT |
| Backend | trtllm | `--backend` | Inference backend used for estimation |
| Prefix Cache Length | 0 | `--prefix` | Prefix cache length for KV reuse |
| Database Mode | SILICON | `--database-mode` | Source of performance data |
| Free GPU Memory Fraction | 1.0 | `--free-gpu-memory-fraction` | KV-cache memory budget used to filter batch sizes |
| Max Sequence Length | isl + osl | `--max-seq-len` | KV blocks TRT-LLM pre-allocates per sequence |
| Chunked Prefill | off | `--enable-chunked-prefill` | Finer-grained context-token sweep |

> **Important:** The TTFT and TPOT defaults act as **SLA filters** — configurations
> that exceed these thresholds are excluded from results. If you see fewer
> results than expected, consider relaxing these values or setting them
> explicitly. When defaults are used, a warning is printed at the start of the
> run so you can verify what values are in effect. By default, only the top-N
> picking step filters on TPOT; pass `--strict-sla` to also pre-filter the
> Pareto frontier (see [Strict SLA filtering](#strict-sla-filtering---strict-sla)).

### Generate mode (Quick Start)
This mode generates a working configuration without running the full parameter sweep. It's useful when you want a quick deployment config without SLA optimization.

```bash
aiconfigurator cli generate --model-path Qwen/Qwen3-32B-FP8 --total-gpus 8 --system h200_sxm
```

The `generate` mode calculates the smallest tensor parallel (TP) size that fits the model in memory using the formula: `TP * VRAM_per_GPU > 1.5 * model_weight_size`. This ensures the model fits with room for KV cache and activations.

**Required arguments:**
- `--model-path` (alias `--model`): HuggingFace model path (e.g., `Qwen/Qwen3-32B-FP8`) or local path containing `config.json`
- `--total-gpus`: Total GPUs for deployment
- `--system`: System name (`h200_sxm`, `h100_sxm`, `h100_pcie`, `gb200`, `b200_sxm`, `a100_sxm`, `a100_pcie`, `l40s`, `l4`, `a30`, `gb300`)

**Optional arguments:**
- `--backend`: Backend name (`trtllm`, `vllm`, `sglang`). Default: `trtllm`
- `--save-dir`: Directory to save generated artifacts
- `--systems-paths`: Override system YAML/data search paths (comma-separated; `default` maps to the built-in systems path). First match wins for identical system/backend/version.

**Example output:**
```
============================================================
  Naive Configuration Generated Successfully
============================================================
  Model:           Qwen/Qwen3-32B-FP8
  System:          h200_sxm
  Backend:         trtllm (1.2.0rc5)
  Total GPUs:      8 (using 8)
  Parallelism:     TP=1, PP=1
  Replicas:        8 (each using 1 GPUs)
  Max Batch Size:  128
  Output:          ./output/Qwen_Qwen3-32B-FP8_naive_tp1_pp1_123456
============================================================
```

**Python API equivalent:**
```python
from aiconfigurator.cli import cli_generate

result = cli_generate(
    model_path="Qwen/Qwen3-32B-FP8",
    total_gpus=8,
    system="h200_sxm",
    backend="trtllm",
    save_dir="./output",
)
print(result["parallelism"])  # {'tp': 1, 'pp': 1, 'replicas': 8, 'gpus_used': 8}
```

> **Note:** This is a naive configuration without memory validation or performance optimization. For production deployments, use `aiconfigurator cli default` to run the full parameter sweep with SLA optimization.

### Estimate mode
This mode runs a single-point performance estimation to predict TTFT (time to first token), TPOT (time per output token), and power consumption for a given model, system, and configuration. Unlike `default` mode, no parameter sweep or SLA optimization is performed — you specify the exact configuration and get back the predicted metrics.

```bash
aiconfigurator cli estimate --model-path Qwen/Qwen3-32B --system h200_sxm --tp-size 2 --batch-size 64 --isl 2048 --osl 512
```

**Required arguments:**
- `--model-path` (alias `--model`): HuggingFace model path (e.g., `Qwen/Qwen3-32B`) or local path containing `config.json`
- `--system`: System name (`h200_sxm`, `h100_sxm`, `h100_pcie`, `b200_sxm`, `gb200`, `a100_sxm`, `a100_pcie`, `l40s`, `l4`, `a30`, `gb300`)

**Optional arguments (shared):**
- `--estimate-mode`: `agg` (default, IFB) or `disagg` (separate prefill/decode workers), or one of the single-pass static breakdown modes `static` / `static_ctx` / `static_gen`
- `--backend`: Backend name (`trtllm`, `vllm`, `sglang`). Default: `trtllm`
- `--backend-version`: Backend database version. Default: latest
- `--database-mode`: Database mode (`SILICON`, `HYBRID`, `EMPIRICAL`, `SOL`). Default: `SILICON`
- `--isl`: Input sequence length. Default: `1024`
- `--osl`: Output sequence length. Default: `1024`
- `--batch-size` (alias `--bs`): Batch size (max concurrent requests, used for agg/static). Default: `128`
- `--ctx-tokens`: Context tokens budget for IFB scheduling (agg only). Default: same as ISL
- `--tp-size` (alias `--tp`): Tensor parallelism size. Default: `1`
- `--pp-size` (alias `--pp`): Pipeline parallelism size. Default: `1`
- `--attention-dp-size` (alias `--dp`): Attention data parallelism size. Default: `1`
- `--moe-tp-size` (alias `--etp`): MoE tensor parallelism size (auto-inferred if omitted)
- `--moe-ep-size` (alias `--ep`): MoE expert parallelism size (auto-inferred if omitted)
- `--gemm-quant-mode`: GEMM quantization mode (auto-inferred from model config if omitted)
- `--kvcache-quant-mode`: KV cache quantization mode (auto-inferred if omitted)
- `--fmha-quant-mode`: FMHA quantization mode (auto-inferred if omitted). DeepSeek-V3 / Kimi-K2.5 context attention (MLA prefill) has no fp8 FMHA data — auto-inferred fp8 is downgraded to `bfloat16` for context-touching estimates (agg, prefill, static, static_ctx, AFD prefill), while decode/generation keeps fp8. Explicitly passing `fp8` for a context estimate is rejected with a clear error.
- `--moe-quant-mode`: MoE quantization mode (auto-inferred if omitted)
- `--comm-quant-mode`: Communication quantization mode (auto-inferred; default `half`)
- `--prefix`: Prefix cache length (subset of ISL already cached per request). Default: `0`
- `--nextn`: MTP draft length, or `auto` to use the checkpoint's `num_nextn_predict_layers` (absent/0 keeps MTP disabled). Default: `0`; MTP is never enabled implicitly — pass `--nextn` explicitly to model it
- `--nextn-accepted`: Average accepted draft tokens per decode step (`0 <= nextn_accepted <= nextn`). Required when the draft depth is > 0 (including via `--nextn auto`); use a measured value from your deployment
- `--stride`: (static modes only) OSL-sweep stride used by `run_static`; ignored for `agg`/`disagg`. Default: `32`
- `--free-gpu-memory-fraction`: Fraction of free GPU memory for KV cache. Default: `0.9`. Used to estimate max concurrent sequences and warn when batch size exceeds KV cache capacity
- `--max-seq-len`: TRT-LLM `--max_seq_len` (default: `isl + osl`). Controls KV blocks pre-allocated per sequence; set to match your deployment for an accurate KV-capacity warning
- `--detail`: Comma-separated breakdown sections to print after the summary box. Choices: `summary`, `memory`, `time`, `energy`, `source`, `all`. Example: `--detail memory,time`. Default: none (use `all` to print every section)

> Shared flags such as `--save-dir`, `--top-n`, and `--systems-paths` are listed in [Common Arguments](#common-arguments-all-modes).

**Disagg-specific arguments** (used when `--estimate-mode disagg`):
- `--decode-system`: System name for decode workers. Defaults to `--system`
- `--prefill-tp-size`, `--prefill-pp-size`, `--prefill-attention-dp-size`: Prefill parallelism overrides (default to shared args)
- `--prefill-moe-tp-size`, `--prefill-moe-ep-size`: Prefill MoE parallelism overrides
- `--prefill-batch-size`: Prefill batch size (required for disagg)
- `--prefill-num-workers`: Number of prefill workers (required for disagg)
- `--decode-tp-size`, `--decode-pp-size`, `--decode-attention-dp-size`: Decode parallelism overrides (default to shared args)
- `--decode-moe-tp-size`, `--decode-moe-ep-size`: Decode MoE parallelism overrides
- `--decode-batch-size`: Decode batch size (required for disagg)
- `--decode-num-workers`: Number of decode workers (required for disagg)

**Example output (agg):**
```text
============================================================
  Performance Estimate (agg)
============================================================
  Model:            Qwen/Qwen3-32B
  System:           h200_sxm
  Backend:          trtllm (1.2.0rc5)
------------------------------------------------------------
  ISL:              2048
  OSL:              512
  Batch Size:       64
  Context Tokens:   2048
  TP Size:          2
  PP Size:          1
------------------------------------------------------------
  TTFT:             487.990 ms
  TPOT:             29.118 ms
  Request Latency:  15367.492 ms
  Power (per GPU):  0.0 W
------------------------------------------------------------
  tokens/s:         2,153.38
  tokens/s/gpu:     1,076.69
  tokens/s/user:    34.34
  seq/s:            4.214
  Concurrency:      64
  Memory (GPU):     54.55 GB
============================================================
```

**Disagg example:**
```bash
aiconfigurator cli estimate \
  --model-path Qwen/Qwen3-32B --system h200_sxm \
  --estimate-mode disagg --isl 2048 --osl 512 --tp-size 2 \
  --prefill-batch-size 4 --prefill-num-workers 2 \
  --decode-batch-size 64 --decode-num-workers 2
```

**Python API equivalent:**
```python
from aiconfigurator.cli.api import cli_estimate

# Aggregated estimation
result = cli_estimate(
    "Qwen/Qwen3-32B", "h100_sxm",
    batch_size=64, isl=2048, osl=512, tp_size=2,
)
print(f"TTFT: {result.ttft:.2f} ms, TPOT: {result.tpot:.2f} ms")
print(f"Power: {result.power_w:.1f} W")
print(f"Throughput: {result.tokens_per_second_per_gpu:,.2f} tokens/s/gpu")

# Disaggregated estimation
result = cli_estimate(
    "Qwen/Qwen3-32B", "h100_sxm", mode="disagg",
    isl=2048, osl=512, tp_size=2,
    prefill_batch_size=4, prefill_num_workers=2,
    decode_batch_size=64, decode_num_workers=2,
)
```

#### Static estimate modes (`static`, `static_ctx`, `static_gen`)

`agg` and `disagg` model continuous (in-flight) batching. The three `static` modes instead run a **single forward pass** through the model — no IFB scheduling — and report the per-phase latency and memory layout for that one pass. They are the quickest way to see where time and memory go for a given shape and parallelism.

- `static` — one full pass over both phases; reports TTFT (context), TPOT (one decode step), and request latency.
- `static_ctx` — context (prefill) phase only; reports TTFT.
- `static_gen` — generation (decode) phase only; reports TPOT.

`--stride N` (static modes only, default `32`) sets the stride `run_static` uses to accelerate the OSL sweep; it is ignored for `agg`/`disagg`.

```bash
aiconfigurator cli estimate \
  --model-path Qwen/Qwen3-32B --system h200_sxm \
  --estimate-mode static --isl 4096 --osl 1024 --tp-size 4 --batch-size 32
```

> A static estimate is a single-pass breakdown, not a served-throughput number — use `agg`/`disagg` for SLA-driven throughput. If the configuration does not fit in memory, the static report still renders but prints an OOM warning.

#### Breakdown report (`--detail`)

`--detail` prints additional breakdown sections after the summary box, for **any** estimate mode (`agg`, `disagg`, or the static modes). Pass a comma-separated list of sections, or `all`:

| Section | Shows |
|---|---|
| `summary` | High-level latency / throughput recap. |
| `memory` | Per-component memory (weights, kvcache, activations, nccl, others) as a share of GPU capacity, the KV footprint per sequence, and a KV-bound max-batch upper bound. |
| `time` | Per-op latency bars in context → generation order, each op's share of the phase, and (in static modes) a Speed-of-Light (SOL) comparison plus the per-op data source. |
| `energy` | Per-op energy breakdown, when energy data is available for the system. |
| `source` | Per-op data-source attribution — `silicon` (measured), `empirical` (interpolated / formula), or `mixed` — so you can tell which numbers are measured vs estimated. |
| `all` | Every section above. |

`--detail` replaces the removed `--print-per-ops-latency` (the old flag still works as a deprecation alias).

```bash
aiconfigurator cli estimate \
  --model-path Qwen/Qwen3-32B --system h200_sxm \
  --estimate-mode static --isl 4096 --osl 1024 --tp-size 4 --batch-size 32 \
  --detail all
```

Abbreviated output (static modes add the SOL columns and per-op `source` tags shown below; `agg`/`disagg` omit the SOL comparison):

```text
Memory Layout (capacity 141.00 GiB)
  weights        15.256 GiB  ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   10.8%
  kvcache        10.000 GiB  ███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    7.1%
  activations    10.000 GiB  ███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    7.1%
  nccl            0.383 GiB  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    0.3%
  others          3.500 GiB  █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    2.5%
  ----------------------------------------------------------------------
  total          39.139 GiB  ███████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   27.8%  (free 101.861 GiB)

  kvcache/seq    0.3125 GiB (seq_len=5120)
  max batch (KV-bound, same isl/osl) ≈ 357
    note: ignores activation growth with batch; treat as an upper bound.

Latency Summary
  metric                 latency            SOL     SOL%
  ttft               3900.357 ms    2997.913 ms    76.9%
  tpot                 10.970 ms       5.572 ms    50.8%
  request latency   15122.575 ms    8697.917 ms    57.5%

Context phase (total = 3900.357 ms, SOL = 2997.913 ms, SOL% = 76.9%)
  op                            latency            SOL     SOL%  share (%)             source
  ...
  context_qkv_gemm           279.049 ms     222.348 ms    79.7%  █░░░░░░░░░░░    7.2%  [silicon]
  context_attention          334.500 ms     142.303 ms    42.5%  █░░░░░░░░░░░    8.6%  [mixed]
  context_gate_ffn1_gemm    1415.684 ms    1111.741 ms    78.5%  ████░░░░░░░░   36.3%  [silicon]
  context_ffn2_gemm          667.966 ms     555.870 ms    83.2%  ██░░░░░░░░░░   17.1%  [silicon]
  ...

Generation phase (total = 11222.218 ms, SOL = 5700.004 ms, SOL% = 50.8%)
  op                               latency            SOL     SOL%  share (%)             source
  ...
  generation_attention         4470.776 ms    2055.779 ms    46.0%  █████░░░░░░░   39.8%  [silicon]
  generation_gate_ffn1_gemm    2509.475 ms    1803.466 ms    71.9%  ███░░░░░░░░░   22.4%  [silicon]
  generation_ffn2_gemm         1345.669 ms     903.968 ms    67.2%  █░░░░░░░░░░░   12.0%  [silicon]
  ...

Data Source Breakdown (per-op)
  context     silicon=8, empirical=5, mixed=1
  generation  silicon=9, empirical=5
```

### Support mode (optional)
This is an optional pre-flight check to verify whether collected SILICON data supports a
specific model and hardware combination for both aggregated and disaggregated serving
modes. You can skip this and run `cli default` directly. `PASS` rows count as support;
`HYBRID_PASS` rows are reported separately as empirical estimability and do not make the
default SILICON support check pass. For unlisted models, support is determined by a
majority vote of SILICON results for models sharing the same architecture.

```bash
aiconfigurator cli support --model-path Qwen/Qwen3-32B-FP8 --system h200_sxm
```

**Required arguments:**
- `--model-path` (alias `--model`): HuggingFace model path (e.g., `Qwen/Qwen3-32B-FP8`) or local path containing `config.json`
- `--system`: System name (`h200_sxm`, `gb200`, `b200_sxm`, `h100_sxm`, `h100_pcie`, `a100_sxm`, `a100_pcie`, `l40s`, `l4`, `a30`, `gb300`)

**Optional arguments:**
- `--backend`: Filter by specific backend (`trtllm`, `vllm`, `sglang`). Defaults to `trtllm`.
- `--backend-version`: Filter by a specific backend version. Defaults to the latest version found in the support matrix for the given model/architecture/system/backend combination.
- `--systems-paths`: Override system YAML/data search paths (comma-separated; `default` maps to the built-in systems path). First match wins for identical system/backend/version.

**Example output:**
```text
============================================================
  AIC Support Check Results
============================================================
  Model:           Qwen/Qwen3-32B-FP8
  System:          h200_sxm
  Backend:         trtllm
  Version:         0.18.0
------------------------------------------------------------
  Aggregated Support:    YES
  Disaggregated Support: YES
============================================================
```

**Python API equivalent:**
```python
from aiconfigurator.cli import cli_support

agg_supported, disagg_supported = cli_support(
    model_path="Qwen/Qwen3-32B-FP8",
    system="h200_sxm",
    backend="trtllm"
)
print(f"Agg: {agg_supported}, Disagg: {disagg_supported}")
```

### Default mode
This mode is triggered by
```bash
aiconfigurator cli default --model-path Qwen/Qwen3-32B-FP8 --total-gpus 32 --system h200_sxm
or
aiconfigurator cli default --model-path Qwen/Qwen3-32B-FP8 --total-gpus 32 --system h200_sxm --ttft 1000 --tpot 10 --isl 3000 --osl 512 --prefix 0
```
`model_path`, `total_gpus`, `system` are three required arguments to define the problem.
If you want to specify your problem with more details, we allow to define `ttft`, `tpot`, `isl`, `osl` and `prefix`.

#### Additional arguments

Beyond `--ttft`, `--tpot`, `--isl`, `--osl`, and `--prefix`, `default` mode accepts:

- `--decode-system`: System (GPU type) for disagg decode workers. Defaults to `--system`. Use it for heterogeneous prefill/decode (e.g. B200 prefill + H200 decode).
- `--backend-version`: Backend database version. Default: latest.
- `--free-gpu-memory-fraction`: Fraction of free GPU memory TRT-LLM allocates for KV cache (default: `1.0`). Filters batch sizes that would exceed KV cache capacity.
- `--max-seq-len`: TRT-LLM `--max_seq_len` (default: `isl + osl`). Controls how many KV blocks are pre-allocated per sequence; set to match your deployment for accurate KV-capacity filtering.
- `--enable-chunked-prefill`: Enable chunked prefill for a finer-grained context-token sweep. When off (default), the context-token stride is aligned to ISL for faster sweeping.
- `--enable-wideep`: Enable Wide Expert Parallelism (WideEP) for MoE models — EP-only parallelism via the `deepep_moe` backend. Applies to DeepSeek and Qwen3-235B on SGLang.
- `--moe-backend`: Explicit SGLang MoE backend — `deepep_moe` or `megamoe` (use `megamoe` to model DeepSeek-V4 MegaMoE on Blackwell).

**Vision-language inputs** (multimodal models such as Qwen3-VL):

- `--image-height`, `--image-width`: Image dimensions in pixels. Default: `0` (disabled — the request is modeled as text-only).
- `--num-images`: Number of images per request. Default: `1`.

The SLA, precision, and speculative-decoding flags (`--strict-sla`, `--request-latency`, `--inclusive-tpot`, `--nextn`, `--nextn-accepted`, `--database-mode`) have dedicated subsections below. Shared flags such as `--save-dir`, `--top-n`, and `--systems-paths` are described in [Common Arguments](#common-arguments-all-modes).

#### Backend Selection

You can specify which inference backend to use with the `--backend` flag:

```bash
# Use TensorRT-LLM (default)
aiconfigurator cli default --model-path Qwen/Qwen3-32B-FP8 --total-gpus 32 --system h200_sxm --backend trtllm

# Use vLLM (dense models only, currently being evaluated)
aiconfigurator cli default --model-path Qwen/Qwen3-32B-FP8 --total-gpus 32 --system h200_sxm --backend vllm

# Use SGLang (dense and MoE models, currently being evaluated)
aiconfigurator cli default --model-path Qwen/Qwen3-32B-FP8 --total-gpus 32 --system h200_sxm --backend sglang
```

Use `--backend auto` to sweep across all supported backends and compare results side by side.
Both agg and disagg results are merged across backends and the globally optimal configuration
is selected. This is useful for finding the best backend without running separate commands.

The command will create two experiments for the given problem, one is `agg` and another one is `disagg`. Compare them to find the better one and estimates the perf gain.

#### Spica migration

The experimental Spica smart sweeper has moved to Dynamo's standalone
[AI Simulate distribution](https://github.com/ai-dynamo/dynamo/blob/4871677d0c4419070729fced4963164bdb1b5221/docs/components/aisimulate/spica/README.md).
The AIC `--thorough-sweep` and `--thorough-config` flags have been removed. Install
it from a matching Dynamo checkout with `python -m pip install ./aisimulate`, then run Spica
through `python -m spica`.

#### Systems Paths

You can override where system YAMLs and performance data are loaded from using `--systems-paths`.

```bash
aiconfigurator cli default \
  --model-path Qwen/Qwen3-32B \
  --total-gpus 32 \
  --system h200_sxm \
  --systems-paths "default,/opt/aic/systems,/data/aic/systems"
```

- Paths are searched in order.
- Use `default` to include the built-in systems path.
- If the same system/backend/version exists in multiple paths, the first match is used.

The command will print out the result to your terminal with the basic info of the comparison, the pareto curve (the best point is tagged as `x`), 
the worker setup for your reference. Let's split them into sections.

Let's run `aiconfigurator cli default --model-path Qwen/Qwen3-32B-FP8 --total-gpus 32 --system h200_sxm --ttft 1000 --tpot 10 --isl 3000 --osl 512 --prefix 0`
> Note that the result might differ based on different versions of your aiconfigurator.
1. Basic info of the comparison
```
  Input Configuration & SLA Target:
    Model: Qwen/Qwen3-32B-FP8 (is_moe: False)
    Total GPUs: 32
    Best Experiment Chosen: disagg at 913.82 tokens/s/gpu (1.43x better)
  ----------------------------------------------------------------------------
  Overall Best Configuration:
    - Best Throughput: 29,242.24 tokens/s
    - Per-GPU Throughput: 913.82 tokens/s/gpu
    - Per-User Throughput: 123.92 tokens/s/user
    - TTFT: 202.65ms
    - TPOT: 8.07ms
```
This shows that for model `Qwen/Qwen3-32B-FP8` to deploy on 32 H200, if you require your TTFT to be less than 1000ms and TPOT to be less than 10ms, and your problem is isl=3000 osl=512, then disagg will be 1.43x of agg. The target result is shown as Overall Best Configuration.

**Python API equivalent:**
```python
from aiconfigurator.cli import cli_default

result = cli_default(
    model_path="Qwen/Qwen3-32B-FP8",
    total_gpus=32,
    system="h200_sxm",
    ttft=1000,
    tpot=10,
    isl=3000,
    osl=512
)
# Access the DataFrames
print(result.best_configs["disagg"])
```

2. Pareto frontier
```
  Pareto Frontier:
              Qwen/Qwen3-32B-FP8 Pareto Frontier: tokens/s/gpu vs tokens/s/user          
    ┌──────────────────────────────────────────────────────────────────────────┐
2250┤ •• disagg                                                                │
    │ ff agg                                                                   │
    │ xx disagg best                                                           │
    │                                                                          │
1875┤  ff                                                                      │
    │   fff                                                                    │
    │     ff                                                                   │
    │      fff••                                                               │
1500┤         f •••                                                            │
    │         ff   ••••••••                                                    │
    │          ffff       •                                                    │
    │              f       •••••••                                             │
1125┤               ff            •                                            │
    │                ff            ••••                                        │
    │                  ffff            ••••x                                   │
    │                     fff              ••••                                │
 750┤                        fff               •                               │
    │                          ffffff           •                              │
    │                                ffffff      ••                            │
    │                                      fffffff ••••••                      │
 375┤                                             ff    •                      │
    │                                               fffffff•••••••••           │
    │                                                      ffffffffff          │
    │                                                                          │
   0┤                                                                          │
    └┬─────────────────┬──────────────────┬─────────────────┬─────────────────┬┘
     0                60                 120               180              240 
tokens/s/gpu                        tokens/s/user                               
```
Pareto frontier shows the trade-off betwen generation speed `tokens/s/user` and throughput `tokens/s/gpu`. The best points is tagged as `x`. As you want the TPOT to be less than 10ms, which means the generation speed is faster than 1000/10ms = 100 tokens/s/user, then by reading the pareto froniter, you will get the point tagged as x. You can see that, if you want different TPOT, you will have different result. Sometimes, agg will be better than disagg (higher throughput at same tokens/s/user)

3. Worker setup
```
  Deployment Details:
    (p) stands for prefill, (d) stands for decode, bs stands for batch size, a replica stands for the smallest scalable unit xPyD of the disagg system
    Some math: total gpus used = replicas * gpus/replica
               gpus/replica = (p)gpus/worker * (p)workers + (d)gpus/worker * (d)workers; for Agg, gpus/replica = gpus/worker
               gpus/worker = tp * pp * dp = etp * ep * pp for MoE models; tp * pp for dense models (underlined numbers are the actual values in math)

disagg Top Configurations: (Sorted by tokens/s/gpu)
+------+--------------+---------------+--------+-------------+------------------+----------+----------------+------------+----------------+-------------+-------+------------+----------------+-------------+-------+
| Rank | tokens/s/gpu | tokens/s/user |  TTFT  | concurrency | total_gpus(used) | replicas |  gpus/replica  | (p)workers | (p)gpus/worker | (p)parallel | (p)bs | (d)workers | (d)gpus/worker | (d)parallel | (d)bs |
+------+--------------+---------------+--------+-------------+------------------+----------+----------------+------------+----------------+-------------+-------+------------+----------------+-------------+-------+
|  1   |    913.82    |     123.92    | 202.65 |  256(=64x4) |   32 (32=4x8)    |    4     |  8 (=4x1+1x4)  |     4      |    1 (=1x1)    |    tp1pp1   |   1   |     1      |    4 (=4x1)    |    tp4pp1   |   64  |
|  2   |    873.07    |     126.28    | 202.65 |  240(=60x4) |   32 (32=4x8)    |    4     |  8 (=4x1+1x4)  |     4      |    1 (=1x1)    |    tp1pp1   |   1   |     1      |    4 (=4x1)    |    tp4pp1   |   60  |
|  3   |    852.77    |     133.94    | 202.65 | 240(=240x1) |   32 (32=1x32)   |    1     | 32 (=12x1+5x4) |     12     |    1 (=1x1)    |    tp1pp1   |   1   |     5      |    4 (=4x1)    |    tp4pp1   |   48  |
|  4   |    568.51    |     148.82    | 202.65 |  144(=72x2) |   32 (32=2x16)   |    2     | 16 (=4x1+3x4)  |     4      |    1 (=1x1)    |    tp1pp1   |   1   |     3      |    4 (=4x1)    |    tp4pp1   |   24  |
|  5   |    434.77    |     145.12    | 123.20 | 104(=104x1) |   32 (24=1x24)   |    1     | 24 (=4x2+4x4)  |     4      |    2 (=2x1)    |    tp2pp1   |   1   |     4      |    4 (=4x1)    |    tp4pp1   |   26  |
+------+--------------+---------------+--------+-------------+------------------+----------+----------------+------------+----------------+-------------+-------+------------+----------------+-------------+-------+

agg Top Configurations: (Sorted by tokens/s/gpu)
+------+--------------+---------------+--------+-------------+------------------+----------+--------------+-------------+----------+----+
| Rank | tokens/s/gpu | tokens/s/user |  TTFT  | concurrency | total_gpus(used) | replicas | gpus/replica | gpus/worker | parallel | bs |
+------+--------------+---------------+--------+-------------+------------------+----------+--------------+-------------+----------+----+
|  1   |    638.28    |     100.97    | 187.72 |  224(=28x8) |   32 (32=8x4)    |    8     |      4       |   4 (=4x1)  |  tp4pp1  | 28 |
|  2   |    612.50    |     101.49    | 274.98 | 224(=14x16) |   32 (32=16x2)   |    16    |      2       |   2 (=2x1)  |  tp2pp1  | 14 |
|  3   |    594.71    |     108.14    | 149.46 |  192(=24x8) |   32 (32=8x4)    |    8     |      4       |   4 (=4x1)  |  tp4pp1  | 24 |
|  4   |    592.60    |     111.22    | 199.08 |  192(=24x8) |   32 (32=8x4)    |    8     |      4       |   4 (=4x1)  |  tp4pp1  | 24 |
|  5   |    544.17    |     119.83    | 149.25 |  160(=20x8) |   32 (32=8x4)    |    8     |      4       |   4 (=4x1)  |  tp4pp1  | 20 |
+------+--------------+---------------+--------+-------------+------------------+----------+--------------+-------------+----------+----+
```

If you want to reproduce the result we esimated, you need to follow the suggestions here. Take the disagg top1 result as an example.  
We're expecting to achieve 913.82 tokens/s/gpu and 123.92 tokens/s/user with this config.  
We have 1 definition `replica`, it means the number of copies of your xPyD disagg system. Say, here, we have 4 replicas, each replica contains 8 GPUs.  
Each replica has a system of 4 prefill workers and 1 decode workers. Each prefill worker is using tp1pp1 which is 1 GPU per worker; while each decoder worker is using tp4pp1 which is 4 GPU per workers. These workers compose a 4P1D replica with 8 GPUs. As you want to deploy on 32 GPUs, then you will have 4 replicas.  
`bs` is required to be set in framework as it limits the largest batch_size of the worker which is crucial to control the TPOT of the deployment.  
`concurrency` = `concurrency * replicas` Use it to benchmark your deployment on total GPUs. If you only want to benchmark 1 replica, divide it by `replicas`

As this is still a little bit challenging to get the right configs for your deployment, we can further specify `--save-dir DIR` to output all the results here as well as **generate the configs for frameworks automatically**. Here is the output folder structure:
```text
results/Qwen_Qwen3-32B-FP8_h200_sxm_trtllm_isl4000_osl1000_ttft1000_tpot20_904495
├── agg
│   ├── best_config_topn.csv
│   ├── config.yaml
│   ├── pareto.csv
│   ├── top1
│   │   ├── agg
│   │   │   ├── agg_config.yaml
│   │   │   ├── bench_run.sh          # aiperf benchmark sweep script (bare-metal)
│   │   │   ├── k8s_bench.yaml        # aiperf benchmark sweep Job (Kubernetes)
│   │   │   ├── k8s_deploy.yaml
│   │   │   └── node_0_run.sh 
│   │   └── generator_config.yaml
│   ...
├── disagg
│   ├── best_config_topn.csv
│   ├── config.yaml
│   ├── pareto.csv
│   ├── top1
│   │   ├── disagg
│   │   │   ├── bench_run.sh          # aiperf benchmark sweep script (bare-metal)
│   │   │   ├── decode_config.yaml
│   │   │   ├── k8s_bench.yaml        # aiperf benchmark sweep Job (Kubernetes)
│   │   │   ├── k8s_deploy.yaml
│   │   │   ├── node_0_run.sh
│   │   │   └── prefill_config.yaml
│   │   └── generator_config.yaml
│   ...
└── pareto_frontier.png
```
By default, we output the top 5 configs we have found. You can get the configs and scripts to deploy under each experiment's folder. The generated files depend on your `--deployment-target`:
- **Dynamo** (default): `k8s_deploy.yaml` for Kubernetes deployment, plus engine configs (`agg_config.yaml`, `prefill_config.yaml`, `decode_config.yaml`) and run scripts (`node_0_run.sh`)
- **llm-d**: `llm-d-values.yaml` for Helm deployment with the llm-d-modelservice chart
- **FPM V1**: exactly `k8s_deploy.yaml` (a reusable keepalive Pod or LeaderWorkerSet) and `run.sh` (the complete rank-aware vLLM launch)

For benchmarking, see the [Benchmark Artifacts](#benchmark-artifacts) section below. Refer to [deployment guide](dynamo_deployment_guide.md) for Dynamo deployments or the [README llm-d section](../README.md#deploying-to-llm-d-platform) for llm-d deployments.

`--save-dir DIR` allows you to specify more information such as generating the config for a different version of the backend, say estimating the performance using trtllm 1.0.0rc3 but generate config for 1.0.0rc6. This is allowed and feasible. By passing `--generated-config-version 1.0.0rc6` can give you the right result.

**Deployment Target Selection**

Use `--deployment-target` to choose which orchestration platform to deploy to:
- `dynamo-j2` (default): Generates typed Dynamo Kubernetes manifests
- `dynamo-python`: Generates Dynamo Kubernetes manifests using Dynamo's Python config modifiers (requires `dynamo` package)
- `llm-d-helm`: Generates Helm values for the llm-d-modelservice chart
- `llm-d-kustomize`: Generates Kustomize overlays for llm-d modelserver guides
- `fpm`: Generates a reusable Kubernetes resource workload and a complete FPM launch script, `run.sh`

The backend (`--backend trtllm/vllm/sglang`) and deployment target are generally orthogonal choices. Note that TRT-LLM only supports Dynamo platforms. FPM V1 is the exception: it supports only a vLLM single aggregated-worker topology with exactly one worker replica; that worker may span multiple nodes. Router/planner configurations and invalid FPM topologies fail closed.

#### FPM V1 resource workload and run script

Use `--deployment-target fpm` when a collector or agent should reserve Kubernetes resources and execute generated FPM launches in them. A generator config can add tokenized vLLM arguments, concrete environment variables, and resource-workload overlays:

```yaml
WorkerConfig:
  agg_workers: 1
Workers:
  agg:
    extra_cli_args:
      - --scheduler-cls
      - InstrumentedScheduler
      - --benchmark-mode
      - agg
      - --dump-config-to
      - "/results/resolved-config-node{node_rank}.json"
K8sConfig:
  fpm_shared_memory_size: 200Gi
  fpm_resource_labels:
    fpm.nvidia.com/run-id: glm52-example
    fpm.nvidia.com/stage: probe
  worker_extra_pod_spec:
    mainContainer:
      resources:
        requests:
          memory: 448Gi
          ephemeral-storage: 30Gi
  extra_env:
    - name: FPM_RUN_ID
      value: glm52-example
    - name: DYN_FPM_BENCHMARK_OUTPUT_PATH
      value: /results/glm52-example.json
```

`Workers.agg.extra_cli_args` must be a `list[str]`, and the final command must include `--benchmark-mode` with one of `agg`, `prefill`, or `decode`. This runtime option selects the FPM collection phase; it does not change the required single aggregated-worker deployment topology. `K8sConfig.extra_env` accepts concrete `{name, value}` entries only; `valueFrom`, `envFrom`, and Secret-derived values are not supported. The normal rule, mapping, and versioned-template pipeline still builds the base command before the extra arguments are appended. If both `--benchmark-output-path` and `DYN_FPM_BENCHMARK_OUTPUT_PATH` are supplied, they must be identical. If neither is supplied, the generator adds `/results/benchmark.json` to both the command and environment.

`K8sConfig.fpm_shared_memory_size` controls the generated `/dev/shm` `emptyDir` limit, while `K8sConfig.fpm_resource_labels` adds labels to the workload and its Pods. Container memory, ephemeral-storage, and other requests or limits can be supplied under `K8sConfig.worker_extra_pod_spec.mainContainer.resources`; the Generator-resolved per-node GPU count cannot be changed there. Matching user-provided `results` or `dshm` volume-and-mount pairs are preserved instead of replaced.

The generated artifact directory contains only:

```text
artifacts/
├── k8s_deploy.yaml
└── run.sh
```

For a single-node topology, `k8s_deploy.yaml` is a keepalive Pod. For a multinode topology, it is a keepalive `LeaderWorkerSet` whose size and per-node GPU limit come from the resolved topology. Both forms contain the requested image, per-node GPU limit, preserved custom resources, volumes, and mounts, but no engine arguments or engine/FPM environment variables. By default they mount Pod-local `emptyDir` storage at `/results`. `run.sh` contains the environment exports and the complete resolved `python3 -m dynamo.vllm ...` command.

`Workers.agg.gpus_per_worker` is the total GPU count for the one worker replica. When that topology exceeds `NodeConfig.num_gpus_per_node`, the generator emits an LWS and divides the GPU limit across its Pods. The total must divide evenly across the resolved node count, and `TP * PP * DP` must match it. Multinode DP must divide evenly across nodes and uses the `mp` data-parallel backend. The target cluster must have the LeaderWorkerSet API and controller installed before applying a multinode artifact.

For one node, apply the Pod, wait for it to become ready, and stream the script into it:

```bash
kubectl apply -f artifacts/k8s_deploy.yaml
kubectl wait --for=condition=Ready pod/<pod> --timeout=10m
kubectl exec -i <pod> -- bash -s < artifacts/run.sh
```

For multiple nodes, the collector stages inputs first and then starts the same `run.sh` concurrently on every LWS Pod. The script requires the controller-injected `LWS_WORKER_INDEX` and `LWS_LEADER_ADDRESS` values to add the rank, master, headless, or local-DP arguments required by that node. Multinode values passed to `--dump-config-to` must contain `{node_rank}`; the default is `/results/resolved-config-node{node_rank}.json`. This placeholder applies only to `--dump-config-to`, not to environment values or arbitrary CLI arguments. Callers must not pass Generator-owned orchestration options such as `--nnodes`, `--node-rank`, `--headless`, or `--data-parallel-size-local` in `extra_cli_args`. On multinode runs, leave `DYN_FPM_WORKER_ID` unset so the script derives `<FPM_RUN_ID>-node<N>`, unless the collector supplies a distinct value to each process.

With DP greater than one, DP rank 0 uses the configured benchmark output path and later DP ranks use `_dp<N>` before the extension. Each node waits for all results in its local rank range. Under the current FPM schema-v1 contract, a result counts as complete only when it has `status: complete`, `valid: true`, complete zero-skipped coverage, the requested benchmark mode and point phase, and nested FPM samples for the expected DP rank. For `agg`, result points may be `prefill` or `decode`. The script also accepts the legacy schema-v2 `status: passed` plus `config.dp_rank` result used by earlier Phase 1 collectors. A terminal invalid result stops the script. The collector still owns staging the complete runtime bundle (scheduler code, cases, capacity, run-spec, runtime contracts, and validators) on every Pod, strict schema/metric validation, result download/aggregation/evidence, exit coordination, and cleanup.

Every execution starts a new engine and reloads the model. `run.sh` refuses to overwrite any expected benchmark output, so each run must use new paths. With the default `/results` `emptyDir`, results remain only for the lifetime of the Pod. FPM V1 does not keep the engine or GPU-resident model alive between executions. Selecting any other deployment target preserves the existing generator output and behavior.

The current vLLM template matrix tops out at `0.20.1`. You may pass reference `0.24.0`-only flags through `extra_cli_args`, but the generator has not yet validated those flags against a `0.24.0` runtime image.

**Generator Dynamo version** (applies to Dynamo deployments only)
- Use `--generator-dynamo-version 0.7.1` to select the Dynamo release. This affects both the generated backend config version and the default K8s image tag.
- If `--generator-dynamo-version` is not provided, the default is the first entry in `backend_version_matrix.yaml` (currently `1.2.0`).
- If `--generated-config-version` is provided, it overrides the generated backend version, but the default K8s image tag still follows the selected Dynamo version mapping.

Use `--generator-config path/to/file.yaml` to provide ServiceConfig/K8sConfig/DynConfig/WorkerConfig/Workers.<role> sections, or add inline overrides via `--generator-set KEY=VALUE`. Examples:

- `--generator-set ServiceConfig.model_path=Qwen/Qwen3-32B-FP8`
- `--generator-set K8sConfig.k8s_namespace=dynamo \`

#### Rule Plugin Selection
You can switch the generator rule set via `--generator-set rule=benchmark`. This selects a rule plugin folder under `src/aiconfigurator/generator/rule_plugin/`.

- **Default (production)**: if `rule` is not provided, the generator uses the default production rules. These are tuned for deployment (e.g., adjusted max batch size and CUDA graph batch sizes).
- **Benchmark**: `--generator-set rule=benchmark` enables rules designed to align generated configs with AIC sdk results, including:
  - wider CUDA graph batch size coverage to match simulated results
  - stricter max batch size that follows the simulated batch size

You can also define your own rule sets by adding a new folder under `src/aiconfigurator/generator/rule_plugin/` and selecting it with `--generator-set rule=<folder_name>`.

Run `aiconfigurator cli default --generator-help` to print information that is sourced directly from `src/aiconfigurator/generator/config/deployment_config.yaml` and `backend_config_mapping.yaml`. 

The `--generator-help` command supports three section options:
- `--generator-help` or `--generator-help all` (default): Shows both the full deployment schema and the backend parameter mappings
- `--generator-help deploy`: Shows the complete content of `generator/config/deployment_config.yaml` in YAML format, including all sections such as `ServiceConfig.*`, `K8sConfig.*`, `WorkerConfig.*`, etc.
- `--generator-help backend`: Shows only the backend parameter mappings table from `generator/config/backend_config_mapping.yaml`, which maps unified parameter keys (e.g., `kv_cache_free_gpu_memory_fraction`, `kv_cache_dtype`) to backend-specific parameter names for trtllm, vllm, and sglang

You can filter the backend-mapping output to a specific backend using `--generator-help --generator-help-backend BACKEND`, where BACKEND can be `trtllm`, `vllm`, or `sglang`. For example:
- `aiconfigurator cli default --generator-help backend --generator-help-backend sglang`: Shows only sglang-specific parameter mappings
- `aiconfigurator cli default --generator-help backend --generator-help-backend trtllm`: Shows only trtllm-specific parameter mappings

The command exits after printing the help information, so you do not need to provide the required `default` mode arguments (like `--model-path`, `--backend`, etc.) when using this flag.

#### Request latency constraint
`--request-latency <ms>` gives you a single end-to-end SLA on TTFT + TPOT × (OSL − 1). When the flag is set, `default` mode automatically enumerates TTFT/TPOT pairs that satisfy that budget (respecting any explicit `--ttft`, if provided) and only keeps configurations whose estimated request latency stays within the bound. Because the CLI derives TPOT from the request latency target, any `--tpot` argument is ignored in this mode.

- The detailed tables printed for both agg and disagg add a `request_latency` column, and the global Pareto plot flips to “request latency vs tokens/s/gpu” whenever every experiment is operating under this constraint.
- You can still set `--ttft` to reserve more headroom for prefill. Leaving it unset lets the enumerator try multiple TTFT splits automatically.

Example: search for 16x H200 configs that meet a 12s end-to-end budget while capping TTFT at 4s.
```bash
aiconfigurator cli default \
  --model-path Qwen/Qwen3-32B-FP8 \
  --total-gpus 16 \
  --system h200_sxm \
  --backend trtllm \
  --request-latency 12000 \
  --isl 4000 \
  --osl 500 \
  --ttft 4000
```
The summary will highlight the fastest configuration whose estimated request latency is ≤ 12,000 ms and will show the derived TTFT/TPOT pair that satisfied the constraint. The example output,
```
********************************************************************************
*                         AIConfigurator Final Results                         *
********************************************************************************
  ----------------------------------------------------------------------------
  Input Configuration & SLA Target:
    Model: Qwen/Qwen3-32B-FP8 (is_moe: False)
    Total GPUs: 16
    Best Experiment Chosen: disagg at 932.91 tokens/s/gpu (disagg 1.09x better)
  ----------------------------------------------------------------------------
  Overall Best Configuration:
    - Best Throughput: 14,926.50 tokens/s
    - Per-GPU Throughput: 932.91 tokens/s/gpu
    - Per-User Throughput: 57.49 tokens/s/user
    - TTFT: 542.58ms
    - TPOT: 17.39ms
    - Request Latency: 9222.18ms
  ----------------------------------------------------------------------------
  Pareto Frontier:
          Qwen/Qwen3-32B-FP8 Pareto Frontier: tokens/s/gpu_cluster vs request_latency    
      ┌────────────────────────────────────────────────────────────────────────┐
1150.0┤ •• agg                                                                 │
      │ ff disagg                                                              │
      │ xx disagg best                                                         │
      │                                                                        │
 958.3┤                                                                        │
      │                                     ffffffffffffffx                    │
      │                                    f                            •      │
      │                                    f                         •••       │
 766.7┤                                    f                    •••••          │
      │                                   f                •••••               │
      │                                 ff            •••••                    │
      │                               ff         •••••                         │
 575.0┤                             ff     ••••••                              │
      │                           ff     •••                                   │
      │                         ff     ••                                      │
      │                              ••                                        │
 383.3┤                          ••••                                          │
      │                        •••                                             │
      │                   ••••••                                               │
      │                 •••                                                    │
 191.7┤                                                                        │
      │                                                                        │
      │                                                                        │
      │                                                                        │
   0.0┤                                                                        │
      └┬─────────────────┬─────────────────┬────────────────┬─────────────────┬┘
       0               3220              6440             9660            12880 
tokens/s/gpu_cluster                request_latency                             

  ----------------------------------------------------------------------------
  Deployment Details:
    (p) stands for prefill, (d) stands for decode, bs stands for batch size, a replica stands for the smallest scalable unit xPyD of the disagg system
    Some math: total gpus used = replicas * gpus/replica
               gpus/replica = (p)gpus/worker * (p)workers + (d)gpus/worker * (d)workers; for Agg, gpus/replica = gpus/worker
               gpus/worker = tp * pp * dp = etp * ep * pp for MoE models; tp * pp for dense models (underlined numbers are the actual values in math)

agg Top Configurations: (Sorted by tokens/s/gpu)
+------+--------------+---------------+--------+-----------------+--------------+-------------------+----------+--------------+-------------+----------+----+
| Rank | tokens/s/gpu | tokens/s/user |  TTFT  | request_latency | concurrency  | total_gpus (used) | replicas | gpus/replica | gpus/worker | parallel | bs |
+------+--------------+---------------+--------+-----------------+--------------+-------------------+----------+--------------+-------------+----------+----+
|  1   |    852.23    |     46.35     | 937.94 |     11704.26    | 320 (=40x8)  |    16 (16=8x2)    |    8     |      2       |  2 (=2x1x1) |  tp2pp1  | 40 |
|  2   |    748.51    |     49.46     | 711.67 |     10799.77    | 256 (=64x4)  |    16 (16=4x4)    |    4     |      4       |  4 (=4x1x1) |  tp4pp1  | 64 |
|  3   |    742.79    |     50.12     | 735.24 |     10691.50    | 256 (=16x16) |    16 (16=16x1)   |    16    |      1       |  1 (=1x1x1) |  tp1pp1  | 16 |
|  4   |    550.53    |     47.56     | 568.11 |     11060.92    | 192 (=96x2)  |    16 (16=2x8)    |    2     |      8       |  8 (=8x1x1) |  tp8pp1  | 96 |
+------+--------------+---------------+--------+-----------------+--------------+-------------------+----------+--------------+-------------+----------+----+

disagg Top Configurations: (Sorted by tokens/s/gpu)
+------+--------------+---------------+--------+-----------------+--------------+-------------------+----------+----------------+------------+----------------+-------------+-------+------------+----------------+-------------+-------+
| Rank | tokens/s/gpu | tokens/s/user |  TTFT  | request_latency | concurrency  | total_gpus (used) | replicas |  gpus/replica  | (p)workers | (p)gpus/worker | (p)parallel | (p)bs | (d)workers | (d)gpus/worker | (d)parallel | (d)bs |
+------+--------------+---------------+--------+-----------------+--------------+-------------------+----------+----------------+------------+----------------+-------------+-------+------------+----------------+-------------+-------+
|  1   |    932.91    |     57.49     | 542.58 |     9222.18     | 384 (=384x1) |    16 (16=1x16)   |    1     | 16 (=10x1+3x2) |     10     |    1 (=1x1)    |    tp1pp1   |   1   |     3      |    2 (=2x1)    |    tp2pp1   |  128  |
|  2   |    932.91    |     49.29     | 542.58 |     10666.29    | 384 (=192x2) |    16 (16=2x8)    |    2     |  8 (=5x1+3x1)  |     5      |    1 (=1x1)    |    tp1pp1   |   1   |     3      |    1 (=1x1)    |    tp1pp1   |   64  |
|  3   |    818.83    |     43.33     | 326.26 |     11842.68    | 328 (=328x1) |    16 (16=1x16)   |    1     | 16 (=6x2+1x4)  |     6      |    2 (=2x1)    |    tp2pp1   |   1   |     1      |    4 (=4x1)    |    tp4pp1   |  328  |
|  4   |    746.33    |     43.72     | 542.58 |     11955.71    | 496 (=496x1) |    16 (16=1x16)   |    1     | 16 (=8x1+1x8)  |     8      |    1 (=1x1)    |    tp1pp1   |   1   |     1      |    8 (=8x1)    |    tp8pp1   |  496  |
+------+--------------+---------------+--------+-----------------+--------------+-------------------+----------+----------------+------------+----------------+-------------+-------+------------+----------------+-------------+-------+
********************************************************************************
2025-12-01 23:36:41,892 - aiconfigurator.cli.main - INFO - All experiments completed in 1.92 seconds
```

#### Inclusive TPOT reporting (`--inclusive-tpot`)

AIC's TPOT metric is the inter-token latency during the decode phase — it does not include TTFT. Pass `--inclusive-tpot` to report TPOT as `(ttft + tpot × (osl − 1)) / osl`, which spreads the TTFT cost across all output tokens. This matches the end-to-end per-token latency reported by GuideLLM and other benchmarking tools, making predicted values directly comparable to benchmark measurements.

The flag only affects terminal output and saved CSV — SLA filtering always uses inter-token latency.

#### Strict SLA filtering (`--strict-sla`)

By default, the Pareto frontier includes all configurations regardless of whether they meet the `--tpot` (or `--request-latency`) constraint — only the final top-N picking step filters on TPOT. This means `pareto.csv` and the Pareto plot may show configurations that violate your SLA targets.

Pass `--strict-sla` to pre-filter the Pareto frontier so that **only SLA-compliant configurations** are included. When this flag is active:

- Configurations exceeding `--tpot` (or `--request-latency`) are removed *before* the Pareto frontier is computed.
- The resulting `pareto.csv`, Pareto plot, and `best_config_topn` only contain configs that meet the SLA.
- TTFT filtering is already enforced at sweep time by all backends, so `--strict-sla` only adds TPOT / request-latency pre-filtering.

```bash
aiconfigurator cli default \
  --model-path Qwen/Qwen3-32B-FP8 \
  --total-gpus 32 \
  --system h200_sxm \
  --tpot 15 \
  --strict-sla
```

> **Note:** With `--strict-sla`, if no configuration meets the SLA targets, the Pareto frontier and best configs will be empty. Without the flag, the Pareto frontier preserves the full search space and you can still see which configs came closest to meeting the target.

The Python API equivalent accepts a `strict_sla` keyword argument:

```python
from aiconfigurator.cli import cli_default

result = cli_default(
    model_path="Qwen/Qwen3-32B-FP8",
    total_gpus=32,
    system="h200_sxm",
    tpot=15,
    strict_sla=True,
)
```

#### Database Mode

The `--database-mode` argument controls how performance is estimated:

| Mode | Description |
|------|-------------|
| `SILICON` | **(Default)** Uses actual collected silicon data. Most accurate when data is available for your configuration. |
| `HYBRID` | Uses silicon data when available, falls back to SOL+empirical factor when data is missing. Best for exploring configurations that may not have complete silicon data. |
| `EMPIRICAL` | Uses Speed-of-Light (SOL) + empirical correction factors for all estimations. Useful for rough estimates without relying on collected data. |
| `SOL` | Provides theoretical Speed-of-Light time only. Useful for understanding theoretical limits. |

PCIe systems such as `h100_pcie`, `a100_pcie`, `l4`, and `a30` are estimate-only unless you provide measured data with `--systems-paths`. They work with `generate` and with non-SILICON modes (`SOL`, `EMPIRICAL`, `HYBRID`); `SILICON` mode still requires a collected performance database.

Example using hybrid mode:
```bash
aiconfigurator cli default --model-path Qwen/Qwen3-32B-FP8 --total-gpus 32 --system h200_sxm --database-mode HYBRID
```

For exp mode, you can specify `database_mode` in your YAML file:
```yaml
exp_hybrid:
  serving_mode: "agg"
  model_path: "Qwen/Qwen3-32B-FP8"
  system_name: "h200_sxm"
  total_gpus: 8
  database_mode: "HYBRID"
```

Hybrid mode is a quick solution to support new models without modeling the operation and collecting the data. However, please be careful, only `SILICON` mode's result is reproducible. Other modes are for research purpose

#### Speculative Decoding (`--nextn`, `--nextn-accepted`)

These flags enable MTP (Multi-Token Prediction) speculative decoding in the
configuration search. MTP is **never enabled implicitly** — omitting `--nextn`
keeps it off even for models that ship MTP layers (the CLI logs a hint when
the checkpoint declares them):

- `--nextn N` — MTP draft length (compute cost side: extra MTP-layer forward
  plus the wider verify batch; no fixed upper bound). Default: 0 (disabled).
- `--nextn auto` — take the draft depth from the checkpoint's
  `num_nextn_predict_layers` (absent or 0 keeps MTP disabled). Only the depth
  comes from the checkpoint — the acceptance value below is still required,
  because it is a property of your workload, not of the model.
- `--nextn-accepted A` — Average accepted draft tokens per decode step
  (`0 <= nextn_accepted <= nextn`); each step yields `1 + nextn_accepted` output tokens.
  Required whenever the draft depth is > 0 (explicit or via `auto`) — there is
  no built-in acceptance assumption. Use a measured value from your deployment
  (e.g. the engine's reported average acceptance length minus 1).

`nextn` is part of the `aic-core` operation and iteration-cost model.
`nextn_accepted` is a workload assumption applied by the SDK predictor/sweep
after core timing, so acceptance values can be swept without recompiling or
rerunning the `aic-core` engine.

Example:
```bash
aiconfigurator cli default \
  --model Qwen/Qwen3-32B-FP8 --total-gpus 8 --system h200_sxm \
  --nextn 2 --nextn-accepted 1.2

# Depth from the checkpoint, acceptance from your measurements:
aiconfigurator cli default \
  --model deepseek-ai/DeepSeek-V3 --total-gpus 8 --system h200_sxm \
  --nextn auto --nextn-accepted 0.7
```


**Python API equivalent:**
```python
from aiconfigurator.cli import cli_exp

# Run experiments from a YAML file
result = cli_exp(yaml_path="example.yaml")

# Or run experiments from a dictionary
config = {
    "my_exp": {
        "serving_mode": "agg",
        "model_path": "Qwen/Qwen3-32B-FP8",
        "total_gpus": 8,
        "system_name": "h200_sxm"
    }
}
result = cli_exp(config=config)
```

See `src/aiconfigurator/cli/exps/database_mode_comparison.yaml` for an example comparing different database modes.

### Benchmark Artifacts

For non-FPM deployment targets, each `topN` directory includes two benchmark helpers alongside the deployment artifacts when `--save-dir` is used. The FPM target emits only `k8s_deploy.yaml` and `run.sh`, so it does not include these helpers.

- **`bench_run.sh`** -- A shell script for bare-metal benchmarking. It loops over a concurrency array and calls [`aiperf profile`](https://github.com/ai-dynamo/aiperf) for each level. Before running it, make sure the deployed service is reachable at the endpoint printed in the script, and that `aiperf` is installed (`pip install aiperf`). Usage:
  ```bash
  cd results/.../disagg/top1/disagg/
  bash bench_run.sh
  ```

- **`k8s_bench.yaml`** -- A Kubernetes Job manifest that runs the same `aiperf` concurrency sweep inside the cluster. Apply it after the service is up:
  ```bash
  kubectl apply -f results/.../disagg/top1/disagg/k8s_bench.yaml
  ```

**Concurrency sweep.** Both artifacts iterate over a base concurrency list `[1, 2, 8, 16, 32, 64, 128]`. When an estimated concurrency is available from the AIConfigurator run, three additional points are added: the estimate itself and its +/-5% neighbors. This targets the operating point AIConfigurator found optimal.

**Templated values.** The scripts are pre-filled with the model name, tokenizer, ISL/OSL, endpoint URL, and streaming mode from the run that generated them -- no manual editing is needed for the common case.

### Exp mode
If you want to customize your experiment apart from simple command which only compares disagg and agg of a same model, you can use `exp` mode. The command is,
```bash
aiconfigurator cli exp --yaml-path example.yaml
```
> **YAML format:** Experiment YAML uses the flat `Task` schema — every key maps
> 1:1 to a `Task` field, with no `mode:` selector and no `config:` /
> `worker_config:` nesting. See [`example.yaml`](../src/aiconfigurator/cli/example.yaml)
> for the annotated template.
>
> The legacy V1 nested format (`mode` / `config` / `worker_config` /
> `replica_config` / `profiles`) is **deprecated** and only a limited
> compatibility shim remains: V1 YAML still loads, but it is auto-converted to V2
> with a `DeprecationWarning`, and any field with no V2 equivalent is rejected
> (not silently dropped). See
> [`example_v1_deprecated.yaml`](../src/aiconfigurator/cli/example_v1_deprecated.yaml)
> for the old shape. Write all new configs in the flat V2 format below.

An example YAML file looks like this; see the [annotated experiment template](../src/aiconfigurator/cli/example.yaml).  
Let's split the yaml file into several sections.  
1. exps
```yaml
exps:
  - agg_full
  - disagg_full
```
`exps` section selects which experiments to run, in order. If omitted, all top-level experiments are run.

2. A certain exp definition
```yaml
disagg_full:
  serving_mode: disagg            # required
  total_gpus: 32                  # required

  # Workload + SLA (shared across roles)
  isl: 4000                       # input sequence length (default 4000)
  osl: 1000                       # output sequence length (default 1000)
  prefix: 0                       # prefix cache length (default 0)
  ttft: 1000.0                    # target TTFT in ms (default 1000.0)
  tpot: 40.0                      # target TPOT in ms (default 40.0)

  # Speculative decoding: never enabled implicitly; nextn_accepted is required
  # when the draft depth is > 0. nextn: auto takes the depth from the
  # checkpoint's num_nextn_predict_layers (the acceptance value is still yours).
  nextn: 1
  nextn_accepted: 0.85

  # --- Prefill role ---
  prefill_model_path: deepseek-ai/DeepSeek-V3   # required
  prefill_system_name: h200_sxm                 # required
  prefill_backend_name: trtllm                  # trtllm (default) | vllm | sglang
  prefill_enable_wideep: false
  # Quant override (default: inferred from the HF model config)
  prefill_gemm_quant_mode: fp8_block            # fp8 | fp8_block | bfloat16
  prefill_moe_quant_mode: fp8_block             # fp8 | fp8_block | w4afp8 | bfloat16
  prefill_kvcache_quant_mode: bfloat16          # fp8 | int8 | bfloat16
  prefill_fmha_quant_mode: bfloat16             # fp8 | bfloat16
  prefill_comm_quant_mode: half
  # Search space (tp=attention, pp=layers, dp=attention DP, moe_tp/moe_ep=MoE)
  prefill_num_gpu_candidates: [4, 8]
  prefill_tp_candidates: [1, 2, 4, 8]
  prefill_pp_candidates: [1]
  prefill_dp_candidates: [1]
  prefill_moe_tp_candidates: [1]
  prefill_moe_ep_candidates: [1, 2, 4, 8]

  # --- Decode role (model_path must equal the prefill model) ---
  decode_model_path: deepseek-ai/DeepSeek-V3    # required
  decode_system_name: h200_sxm                  # required
  decode_backend_name: trtllm
  decode_enable_wideep: false
  decode_gemm_quant_mode: fp8_block
  decode_moe_quant_mode: fp8_block
  decode_kvcache_quant_mode: bfloat16
  decode_fmha_quant_mode: bfloat16
  decode_comm_quant_mode: half
  decode_num_gpu_candidates: [4, 8]
  decode_tp_candidates: [1, 2, 4, 8]
  decode_pp_candidates: [1]
  decode_dp_candidates: [1, 2, 4, 8]
  decode_moe_tp_candidates: [1]
  decode_moe_ep_candidates: [1, 2, 4, 8]

  # --- Disagg orchestration: replica shaping + perf correction ---
  num_gpu_per_replica: [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128]
  max_gpu_per_replica: 128        # caps num_gpu_per_replica (0 = no limit)
  max_prefill_workers: 32         # max prefill workers per replica (x in xPyD)
  max_decode_workers: 32          # max decode workers per replica (y in xPyD)
  prefill_latency_correction: 1.1
  decode_latency_correction: 1.08
  prefill_max_batch_size: 1
  decode_max_batch_size: 512
```
This is long; the basics:  
    - `serving_mode`: `agg` or `disagg` for this experiment.  
    - `total_gpus`: total GPU budget for the deployment.  
    - For `disagg`, the worker spec is per-role: set `prefill_*` / `decode_*` for `model_path`, `system_name`, `backend_name`, the `*_quant_mode` fields, and the `*_candidates` search lists. `decode_model_path` must equal `prefill_model_path` (hetero-disagg means different *systems*, not models).  
    - For `agg`, the same fields are top-level (`model_path`, `system_name`, `gemm_quant_mode`, `agg_tp_candidates`, ...) — see `agg_full` in the template.  
    - `backend_name`: `trtllm` (default), `vllm`, or `sglang`.  
    - `backend_version`, `isl`, `osl`, `ttft`, `tpot`: same meaning as in `default` mode (shared, top-level).  
    - `*_enable_wideep`: enables wide-EP for fine-grained MoE models.  
    - `nextn` / `nextn_accepted`: MTP speculative decoding (never auto-enabled; `nextn_accepted` is required when the resolved `nextn > 0`).
    - The replica/correction knobs (`num_gpu_per_replica`, `max_*_workers`, `*_latency_correction`, ...) are covered in [Advanced Tuning](advanced_tuning.md). Typically the only thing you need to touch is the quantization.

Quantization override order: explicit `*_quant_mode` fields take precedence; any mode left unset is filled from the model's HF quantization metadata.

You can drop everything optional and keep just the required fields plus the few knobs you care about. Here's a minimal disagg with wide-EP:
```yaml
disagg_simplified:
  serving_mode: disagg
  total_gpus: 512
  nextn: 2
  nextn_accepted: 1.1
  prefill_model_path: deepseek-ai/DeepSeek-V3
  prefill_system_name: gb200
  prefill_enable_wideep: true        # wide-EP for prefill
  decode_model_path: deepseek-ai/DeepSeek-V3
  decode_system_name: gb200
  decode_enable_wideep: true         # wide-EP for decode
  max_gpu_per_replica: 512           # wide-EP needs a larger replica budget
```
Everything omitted falls back to defaults / HF inference.

Let's go through some pre-defined experiments for reference.
1. homegeneous vs. heterogenous  
The example [yaml](../src/aiconfigurator/cli/exps/hetero_disagg.yaml)
```yaml
exps:
  - exp_h200_h200
  - exp_b200_h200

exp_h200_h200:
  serving_mode: disagg
  total_gpus: 16
  isl: 4000
  osl: 500
  ttft: 300.0
  tpot: 50.0
  prefill_model_path: Qwen/Qwen3-32B-FP8
  prefill_system_name: h200_sxm      # prefill on H200
  prefill_backend_name: trtllm       # vllm | sglang also work
  decode_model_path: Qwen/Qwen3-32B-FP8
  decode_system_name: h200_sxm       # decode on H200
  decode_backend_name: trtllm

exp_b200_h200:
  serving_mode: disagg
  total_gpus: 16
  isl: 4000
  osl: 500
  ttft: 300.0
  tpot: 50.0
  prefill_model_path: Qwen/Qwen3-32B-FP8
  prefill_system_name: b200_sxm      # prefill on B200
  prefill_backend_name: trtllm
  decode_model_path: Qwen/Qwen3-32B-FP8
  decode_system_name: h200_sxm       # decode on H200
  decode_backend_name: trtllm
```
We defined two experiments. `exp_h200_h200` uses H200 for both prefill and decode. `exp_b200_h200` uses B200 for prefill and H200 for decode — hetero-disagg is expressed purely by giving the two roles different `*_system_name` values (the model must be the same).

**Note**: You can also compare different backends by setting different `backend_name` values (trtllm, vllm, sglang) in your experiments.

2. use a specific quantization  
The example [yaml](../src/aiconfigurator/cli/exps/qwen3_32b_pertensor.yaml)
```yaml
exps:
  - exp_agg
  - exp_disagg

exp_agg:
  serving_mode: agg
  model_path: Qwen/Qwen3-32B-FP8
  system_name: h200_sxm
  total_gpus: 16
  backend_name: trtllm
  isl: 4000
  osl: 500
  ttft: 600.0
  tpot: 16
  # per-tensor FP8 on every component
  gemm_quant_mode: fp8
  moe_quant_mode: fp8
  kvcache_quant_mode: fp8
  fmha_quant_mode: fp8
  comm_quant_mode: half

exp_disagg:
  serving_mode: disagg
  total_gpus: 16
  isl: 4000
  osl: 500
  ttft: 600.0
  tpot: 16
  prefill_model_path: Qwen/Qwen3-32B-FP8
  prefill_system_name: h200_sxm
  prefill_backend_name: trtllm
  prefill_gemm_quant_mode: fp8
  prefill_moe_quant_mode: fp8
  prefill_kvcache_quant_mode: fp8
  prefill_fmha_quant_mode: fp8
  prefill_comm_quant_mode: half
  decode_model_path: Qwen/Qwen3-32B-FP8
  decode_system_name: h200_sxm
  decode_backend_name: trtllm
  decode_gemm_quant_mode: fp8
  decode_moe_quant_mode: fp8
  decode_kvcache_quant_mode: fp8
  decode_fmha_quant_mode: fp8
  decode_comm_quant_mode: half
```
Here we override the quantization of Qwen/Qwen3-32B-FP8: the default is blockwise FP8 for GEMM, and we set per-tensor FP8 explicitly via the `*_quant_mode` fields. (The deprecated V1 way was `profiles: ["fp8"]`, which expanded to exactly these fields.)

You can refer to [src/aiconfigurator/cli/exps](../src/aiconfigurator/cli/exps) to find more reference yaml files.

Use `exp` mode for flexible experiments, `default` mode for convenient agg vs disagg comparison with SLA optimization, and `generate` mode for quick config generation without sweeping. All modes support generating configs for frameworks automatically by `--save-dir DIR`.

---


## End-to-End Workflow

This section walks through the typical workflow from checking hardware/model support all the way to benchmarking a deployed service. Each step feeds into the next.

**Scenario:** Deploy Qwen3-32B-FP8 on 8x H200 GPUs with SLA targets of TTFT <= 600 ms and TPOT <= 50 ms.

### Step 1: Check support (optional)

You can optionally verify that your model/system combination is supported before running a sweep. This step is not required — you can skip it and run `cli default` directly.

```bash
aiconfigurator cli support --model Qwen/Qwen3-32B-FP8 --system h200_sxm
```

If the output shows `Aggregated Support: YES` and/or `Disaggregated Support: YES`, proceed. Otherwise, try a different backend (`--backend vllm` or `--backend sglang`) or system.

### Step 2: Find the optimal configuration

Run the parameter sweep to compare aggregated vs. disaggregated serving and find the best config under your SLA:

```bash
aiconfigurator cli default \
  --model Qwen/Qwen3-32B-FP8 \
  --total-gpus 8 \
  --system h200_sxm \
  --ttft 600 --tpot 50 \
  --isl 4000 --osl 500 \
  --save-dir results \
  --generator-set ServiceConfig.head_node_ip=0.0.0.0 \
  --generator-set ServiceConfig.model_path=/workspace/models/Qwen3-32B-FP8
```

`--save-dir` generates deployment-ready artifacts (engine configs, run scripts, K8s manifests, and benchmark helpers) under `results/`.

### Step 3: Quick fallback (optional)

If `cli support` shows your model/system combo is unsupported, or `cli default` fails to find a valid configuration, `generate` gives you the smallest TP that fits the model in memory. Otherwise, you can use the `cli default` results directly and skip this step.

```bash
aiconfigurator cli generate \
  --model Qwen/Qwen3-32B-FP8 \
  --total-gpus 8 \
  --system h200_sxm \
  --save-dir results_naive
```

### Step 4: Deploy

Use the generated artifacts to launch the service. For bare-metal (single-node):

```bash
mkdir -p /workspace/engine_configs
cp results/.../disagg/top1/disagg/*_config.yaml /workspace/engine_configs/
cd results/.../disagg/top1/disagg/
bash node_0_run.sh
```

For Kubernetes (Dynamo):

```bash
kubectl apply -f results/.../disagg/top1/disagg/k8s_deploy.yaml
```

For llm-d (Helm):

```bash
helm install my-model llm-d/llm-d-modelservice \
  --values results/.../disagg/top1/llm-d-values.yaml
```

See the [Deployment Guide](dynamo_deployment_guide.md) for multi-node and K8s details.

### Step 5: Benchmark

After the service is healthy, run the generated benchmark sweep to validate performance at the predicted concurrency:

```bash
# Bare-metal
bash results/.../disagg/top1/disagg/bench_run.sh

# Or Kubernetes
kubectl apply -f results/.../disagg/top1/disagg/k8s_bench.yaml
```

Compare the measured TTFT, TPOT, and tokens/s/gpu against the AIConfigurator estimates printed in Step 2. See [Benchmark Artifacts](#benchmark-artifacts) for details on the generated scripts.
