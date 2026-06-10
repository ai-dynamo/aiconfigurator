---
name: run-nsys-in-vllm-container
description: Run Nsight Systems inside a vLLM Docker container for layerwise or FPM profiling. Use when Docker image lacks nsys, when mounting host Nsight Systems into vllm/vllm-openai or Dynamo vLLM containers, or when validating that exported nsys sqlite files contain CUDA/CUPTI kernel traces rather than NVTX-only spans.
---

# Run Nsys In vLLM Container

## When The Image Lacks `nsys`

First check inside the target container:

```bash
docker run --rm --entrypoint /bin/bash IMAGE -lc 'command -v nsys || true; nsys --version || true'
```

If missing, mount host Nsight Systems and expose both the CLI and target libraries:

```bash
NSYS_HOME=/opt/nvidia/nsight-systems/2025.6.3
docker run --rm --gpus '"device=0"' --ipc=host \
  --entrypoint /bin/bash \
  -e PATH="$NSYS_HOME/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  -e LD_LIBRARY_PATH="$NSYS_HOME/target-linux-x64:${LD_LIBRARY_PATH:-}" \
  -v "$NSYS_HOME:$NSYS_HOME:ro" \
  -v "$PWD:/work/aiconfigurator" \
  -w /work/aiconfigurator \
  IMAGE \
  -lc 'nsys --version && python3 collector/layerwise/vllm/collect_layerwise.py ...'
```

On this host, `/usr/local/cuda/bin/nsys` is only a wrapper. Prefer mounting the real Nsight tree under `/opt/nvidia/nsight-systems/<version>`.

## Validation

For AIC layerwise data, `nsys` must collect CUDA/CUPTI tables. NVTX-only traces are not valid even if the collector writes a CSV row.

After a run, inspect `profiles/status.jsonl`:

```bash
rg '"nsys_parse_succeeded"|"success"' RUN_DIR/profiles/status.jsonl
```

Valid layerwise traces should show:

- `meta.attribution_source` is `cupti`
- `meta.attributed_kernels` is nonzero
- success rows have nonzero `kernel_count`
- exported sqlite is usually much larger than an NVTX-only trace for the same run

If `meta.attribution_source` is `nvtx_span`, treat the latency row as invalid for layerwise kernel attribution. That means Nsight saw NVTX ranges but did not record CUDA kernel/runtime tables.

On this host, prefer `/opt/nvidia/nsight-systems/2025.6.3` for CUDA 12.9 containers such as `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.0`. A smoke test with `/opt/nvidia/nsight-systems/2024.6.2` and `--capture-range=cudaProfilerApi` fired `cudaProfilerStart/Stop` but finalized with an empty `Generated:` line and no `.nsys-rep`; the same test with 2025.6.3 produced a valid report. Mounting 2024.6.2 into `vllm/vllm-openai:v0.20.1` made `nsys` runnable, but both with and without `LD_LIBRARY_PATH=$NSYS_HOME/target-linux-x64` the exported sqlite stayed NVTX-only. In these cases, do not use the CSV latency for layerwise data. Prefer a container/image with native Nsight/CUPTI support, or keep iterating on container privileges/CUPTI injection until the validation shows `attribution_source=cupti`.

## Notes For This Repo

- The vLLM collector launches `nsys profile` from inside the scheduler process, so `nsys` must be visible inside the container that runs `collector/layerwise/vllm/collect_layerwise.py`.
- Keep the collector CLI short; common vLLM extras such as `--skip-mm-profiling`, `--limit-mm-per-prompt {"image":0,"video":0}`, and `--generation-config vllm` are added by the collector.
- Do not expose HF tokens in process listings. Prefer `HF_TOKEN_FILE=~/hf.token` in wrapper scripts, or pass token environment variables without printing `ps -ef` command lines.
