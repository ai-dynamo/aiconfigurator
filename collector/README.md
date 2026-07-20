<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# Introduction
Data collection is a standalone process for collecting the database for aiconfigurator. By default, you don't have to collect the data by yourself.
Small versions of database will not introduce huge perf difference. Say, you can use 1.0.0rc3 data of trtllm on h200_sxm and deploy the generated
configs with Dynamo + trtllm 1.0.0rc4 worker.

If you want to go through the process, you can try belowing commands. However, you need to prepare the env by yourself such as installing a specific trtllm version.
This process is not well verified, you need to debug sometimes.

For a framework-version or GPU-platform upgrade, follow the
[Collector Upgrade Playbook](../docs/perf_database/collector-upgrade-playbook.md).
The repo-tracked [`aic-auto-collect`](../.claude/skills/aic-auto-collect/SKILL.md)
skill applies that workflow during long, resumable collection runs.
SGLang 0.5.14 Hopper/Blackwell follow-up work must also consult the
platform alignment ledger, a local campaign record kept outside the repo
(`docs/perf_database/sglang-0.5.14-hopper-blackwell-ledger.md` in the
campaign workspace).

# Preparation
Before collecting the data, make sure you own the whole node and no interfierence happens.
Next, please enable persistent-mode and lock frequency of the node. Make sure the cooling system of the node is working well.
```bash
sudo nvidia-smi -pm 1
```
```bash
sudo nvidia-smi -ac yyy,xxx
```
xxx, yyy frequency can be queried by nvidia-smi -q -i 0, refer to the Max Clocks part, xxx is SM frequency, yyy is Memory frequency.
A script to set frequency:
```
#!/bin/bash

# Run nvidia-smi query and extract SM and Memory frequencies from Max Clocks
sm_freq=$(nvidia-smi -q -i 0 | grep -A 4 "Max Clocks" | grep "SM " | grep -o "[0-9]\+ MHz" | grep -o "[0-9]\+")
mem_freq=$(nvidia-smi -q -i 0 | grep -A 4 "Max Clocks" | grep "Memory " | grep -o "[0-9]\+ MHz" | grep -o "[0-9]\+")

# Check if frequencies were successfully extracted
if [ -z "$sm_freq" ] || [ -z "$mem_freq" ]; then
    echo "Error: Could not extract SM or Memory frequency from Max Clocks."
    exit 1
fi

# Generate the command
echo "sudo nvidia-smi -ac $mem_freq,$sm_freq"
```
Prepare a clean env with the target framework and nccl lib installed.

# Collect comm data
```bash
export PATH=$PATH:${NCCL_TEST_BIN_PATH}/
network/collect_comm.sh #all_reduce data will be collected using default trtllm backend
network/collect_comm.sh --all_reduce_backend vllm #all_reduce data will be collected using vllm backend
network/collect_comm.sh --all_reduce_backend vllm --device xpu #all_reduce data will be collected using vllm backend on XPU
```
Today we only collect intra-node comm. This script will collect custom allreduce data for trtllm within a node.
It will also collect nccl allreduce, all_gather, all2all, reduce_scatter using nccl.
The generated files are nccl_perf.txt, oneccl_perf.txt, and custom_allreduce_perf.txt.

# Model-centric cases and healing runs

The model-centric case layer (introduced as "collector v2") is how collection
coverage is declared, and it is unchanged under Collector V3 — V3 adds
op-centric *management* on top (per-family runtime pins, provenance sidecars,
declared reuse; see `docs/perf_database/collector-v3-op-centric-design.md`),
it does not replace how cases are declared or healed. A healing run collects
cases for a specific model/GPU pair instead of running every op bucket and
hoping the support matrix improves. Use `--model-cases-full` when you want
the case YAML to define a full model-centric run. Omitting all model-case
flags runs the backend registry directly without a model-centric case plan.

```bash
# Heal one model on one GPU type.
python3 collect.py --backend sglang \
  --model-path sgl-project/DeepSeek-V4-Flash-FP8 \
  --gpu b200_sxm

# Inspect the resolved model plan without collecting.
python3 collect.py --backend sglang \
  --model-path sgl-project/DeepSeek-V4-Flash-FP8 \
  --gpu b200_sxm \
  --plan-only

# Select the architecture case file directly.
python3 collect.py --backend trtllm \
  --model-architecture Qwen3MoeForCausalLM \
  --gpu b200_sxm \
  --plan-only

# Full model-centric run: aggregate base ops plus every model case YAML file.
python3 collect.py --backend trtllm --model-cases-full

# Raw backend registry run: no model-centric case plan.
python3 collect.py --backend trtllm

# Ephemeral subset filter (healing): substring match, repeatable.
python3 collect.py --backend trtllm --ops moe --case-filter "tp=4"
```

Case files (see `cases/README.md` for the full authoring guide):

```text
cases/base_ops/<op>.yaml               — shared sweep recipes (interpolation grid)
cases/models/<architecture>_cases.yaml — op activation + model shape values
cases/capabilities.yaml                — positive hardware floors (dtype/op -> min SM)
cases/denylist.yaml                    — hang/node-killer cases only
model_cases.py                         — resolves which ops a model plan collects
capabilities.py                        — generation-time capability/denylist filter
```

The plan is one equation: cases = dedup(base grid ∪ model shapes), then
intersected with hardware capability floors and minus the hang denylist.

## Failure philosophy: observe, don't predict

There is no declarative expected-failure layer. A case that cannot run on the
current framework version fails fast and is recorded in `errors_<module>.json`
and `collection_summary_<backend>.json` with a classification and a
(model, dtype) group label. The summary aggregates failures by group: a whole
group failing is a fix-me signal (collector bug, unverified combo, framework
gap), not something to tolerate or auto-skip. Missing points are tolerated
downstream by interpolation and fallback. Escalation rules — when a failure
deserves a denylist entry, a registry `unverified` marker, or a capability
floor — live in `.claude/rules/collector/failure_handling.md`.

`--gpu <type>` resolves the SM version from
`src/aiconfigurator/systems/<gpu>.yaml`; use `--sm <version>` on an
unregistered GPU. Without either, the local device capability is detected
automatically.

To add a new architecture, create one `cases/models/<architecture>_cases.yaml`
file. To add a new model in an existing architecture, add the model path to
that architecture's `model_paths` list. Add shared op sweeps to the matching
`cases/base_ops/<op>.yaml` file. Add a new op collector only when the existing
ops cannot generate the needed data points.

# Version Management

## Overview

Each backend (trtllm, vllm, sglang) has a **registry** (`registry.py`) that maps ops to collector modules, and a **version resolver** (`version_resolver.py`) that picks the right module at runtime. Individual collector files declare their compatibility via `__compat__`. The current collector framework versions and runtime images are declared in `framework_manifest.yaml`.

```text
framework_manifest.yaml — current collector framework versions and images (schema v2)
framework_manifest.py   — manifest loader/validator/resolver
op_catalog.py         — table→family identity map (op_backend_catalog.yaml)
model_cases.py       — model-centric model/SM case-plan resolver
registry.py          — declares which module handles which version range
version_resolver.py  — routes runtime version → module (packaging.version)
collect.py/collect_ops — validates __compat__ and fails incompatible ops
__compat__           — per-file metadata declaring supported framework versions
cases/               — model-centric case manifests and SM exceptions
wideep/              — WideEP collector namespace for special images/runtimes
wideep/*/registry.py — WideEP-only ops appended when the v2 plan requests them
network/             — collective communication collectors and Slurm network jobs
```

`framework_manifest.yaml` is `schema_version: 2`: each framework entry has a
`default` runtime (`version` + `images`) plus an optional `families:` map of
per-family overrides. Resolution for a given `(framework, op)` walks
`table → catalog family → families[family] or default` — exactly one runtime
per op, always. `families:` overrides require the op catalog
(`op_backend_catalog.yaml`); without it, family-scoped pins fail closed rather
than silently falling back to `default`.

WideEP runtimes (`wideep_sglang`, `wideep_trtllm`) are flattened into peer
framework entries rather than nested under a `wideep:` key, so the same
per-op resolution logic applies uniformly. Each WideEP entry adds
`base_framework` (the stock framework it forks from), `data_backend` (which
table namespace its output is written under), and `collector_dir`.
`collect.py` requires the exact public version for a run and rejects
mixed-pin selections across ops.

Image digests are policy, not decoration: the rule is syntactic — any image
reference containing `/` must be pinned with `@sha256:<64 hex>`; only bare
internal image names (no `/`, e.g. `deepseek-v4-blackwell`) are exempt. In
practice that means every public-registry reference is digest-pinned.
When bumping a version pin, re-fetch and update the digest for every image
variant touched — a stale digest silently keeps pulling the old image. Always
pin the manifest-list (index) digest, never a platform-child digest —
`docker buildx imagetools inspect <image>` prints the index digest; a child
digest pins one architecture and breaks the other platforms.

Use `collector.framework_manifest.resolve_op_runtime(framework, op)` to
resolve a single op's runtime, and
`collector.framework_manifest.validate_resolution()` to check the whole
manifest against every backend registry at once — it returns a list of error
strings (empty = valid) and is the fail-closed CI gate asserting every
registered op resolves to exactly one pinned runtime. See
`docs/perf_database/collector-v3-op-centric-design.md` §4 for the full
design.

## File Naming Convention

- **No version fork**: `collect_{op}.py` (no suffix)
- **Version forks**: `collect_{op}_v1.py`, `collect_{op}_v2.py`, ... (sequential)
- The number is just an ordering key; the actual version range lives in `__compat__`

## `__compat__` Format

Every collector file must declare `__compat__` after the license header:

```python
# SPDX-FileCopyrightText: ...
# SPDX-License-Identifier: Apache-2.0

__compat__ = "trtllm>=1.1.0"
```

The format is `<framework><constraints>` using PEP 440-style operators:
- `"trtllm>=1.1.0"` — open-ended (latest version)
- `"trtllm>=0.21.0,<1.1.0"` — bounded range
- `"sglang>=0.5.5"` — sglang example

Pre-release ordering is respected: `1.1.0rc2 < 1.1.0 < 1.1.0.post1`.

## Registry Format

Each entry in `registry.py` is an `OpEntry` dataclass (defined in `collector/registry_types.py`).
Exactly one of `module` (unversioned) or `versions` (versioned) must be provided — this is
validated at construction time.

```python
from collector.registry_types import OpEntry, VersionRoute

# Unversioned (no fork):
OpEntry(op="gemm", module="collector.trtllm.collect_gemm", get_func="...", run_func="...")

# Versioned (has forks) — VersionRoutes in descending min_version order:
OpEntry(op="myop", get_func="...", run_func="...", versions=(
    VersionRoute("X.Y.Z", "collector.<backend>.collect_myop_v2"),
    VersionRoute("0.0.0", "collector.<backend>.collect_myop_v1"),
))
```

The resolver picks the first `VersionRoute` where `min_version <= runtime_version`.

## Adding a New Op

1. Create `collector/<backend>/collect_myop.py`
2. Add `__compat__ = "<backend>>=X.Y.Z"` after the license header
3. Export `get_myop_test_cases()` and `run_myop()` (or `run_myop_torch()`)
4. Add an entry to `collector/<backend>/registry.py`:
   ```python
   OpEntry(op="myop", module="collector.<backend>.collect_myop", get_func="get_myop_test_cases", run_func="run_myop")
   ```
5. Run `pytest tests/unit/collector/ -m unit` to verify registry integrity

## Handling a Framework API Change

When upstream framework `X.Y.Z` changes an API that a collector depends on:

1. Rename the original file to `collect_{op}_v1.py` (if not already versioned)
2. Create `collect_{op}_v2.py` with the new API calls
3. Add `__compat__ = "<backend>>=X.Y.Z"` to the new file
4. Add `__compat__` upper bound to the old file if it doesn't have one (e.g. change `">=1.1.0"` to `">=1.1.0,<X.Y.Z"`)
5. Convert the registry entry from unversioned to versioned (or prepend a new `VersionRoute`):
   ```python
   # Before (unversioned):
   OpEntry(op="gemm", module="collector.trtllm.collect_gemm", ...)

   # After (versioned — all forks carry explicit _vN suffix):
   OpEntry(op="gemm", versions=(
       VersionRoute("X.Y.Z", "collector.<backend>.collect_gemm_v2"),
       VersionRoute("0.0.0", "collector.<backend>.collect_gemm_v1"),
   ), ...)
   ```
6. Run tests to validate

## Runtime Behavior

At collection time:
1. `collect.py` reads the backend registry
2. `build_collections()` resolves each op to the correct module for the detected framework version
3. If the resolved module declares `__compat__`, it is validated against the runtime version — mismatches fail that op explicitly and are recorded in the error summary
4. Unsupported versions (no matching entry) are skipped with a warning

## Tests

```bash
pytest tests/unit/collector/ -m unit
```

Covers: version parsing (PEP 440 with rc/post), `__compat__` constraint evaluation, module routing, and structural validation of all three registries.

# Collect gemm/attention/moe data/etc.

## Smoke Test

Use `--smoke` to quickly verify the collector runs end-to-end. It randomly samples 4 test cases per op instead of running the full suite.

```bash
# Smoke test for sglang
python3 collect.py --backend sglang --smoke

# Smoke a specific op
python3 collect.py --backend trtllm --ops moe --smoke
```

## Power Monitoring (Optional)

The collector supports GPU power monitoring during kernel execution using NVML. This feature is optional and disabled by default.

### Enable Power Monitoring
```bash
# Basic power monitoring
python3 collect.py --backend trtllm --measure_power

# With custom minimum duration (default: 1.0s)
python3 collect.py --backend trtllm --measure_power --power_test_duration_sec 2.0
```

### Options
- `--measure_power`: Enable NVML-based power monitoring (samples at 100ms intervals)
- `--power_test_duration_sec`: Minimum test duration for accurate power readings (default: 1.0s)

### Output
When power monitoring is enabled, performance CSV files will include additional columns:
- `power`: Average power consumption during kernel execution (Watts)
- `power_limit`: GPU power management limit (Watts)

**Example output:**
```csv
framework,version,device,op_name,kernel_source,gemm_dtype,m,n,k,latency,power,power_limit
TRTLLM,1.2.0,NVIDIA H200 SXM,gemm,torch_flow,bfloat16,1024,4096,4096,0.234,523.4,700.0
```

### Requirements
Power monitoring requires:
- `pynvml` Python package: `pip install pynvml`
- NVML support (NVIDIA drivers)

If unavailable, a warning is logged and execution continues without power data.

### Notes
- Power monitoring adds minimal overhead (<1%)
- Kernel iterations are automatically adjusted to meet minimum duration for accurate measurements
- Backward compatible: without `--measure_power`, CSVs remain unchanged

## CUDA Graph Fallback Support

The `benchmark_with_power` helper function now supports graceful fallback to eager execution when CUDA graph capture fails. This is particularly useful for complex operations like MOE (Mixture of Experts) with large batch sizes.

### Features
- **Automatic fallback**: When `allow_graph_fail=True`, CUDA graph capture failures trigger eager execution instead of raising exceptions
- **Power measurement in both paths**: Power monitoring works correctly in both graph replay and eager execution modes
- **Memory safety**: Automatic `torch.cuda.empty_cache()` call on graph capture failure to prevent memory fragmentation
- **Transparency**: Results include `used_cuda_graph` flag to indicate which execution path was used

### Usage Example
```python
from helper import benchmark_with_power

def my_kernel():
    # Your kernel code here
    moe.forward(hidden_states, logits)

# Use benchmark_with_power with fallback support
with benchmark_with_power(
    device=device,
    kernel_func=my_kernel,
    num_warmups=3,
    num_runs=6,
    repeat_n=1,
    allow_graph_fail=True,  # Enable graceful fallback
) as results:
    latency = results["latency_ms"]
    power_stats = results["power_stats"]  # Available in both paths

    # Check which execution path was used
    if not results["used_cuda_graph"]:
        print("CUDA graph capture failed, used eager execution")
```

### When to Use
- **Complex operations**: MOE, dynamic memory patterns, or operations that may not be graph-compatible
- **Large batch sizes**: When graph capture may fail due to memory constraints
- **Development/debugging**: To ensure collection continues even if graph capture fails

### Backward Compatibility
- Default behavior unchanged: `allow_graph_fail=False` maintains existing behavior
- Existing collectors work without modifications
- Only opt-in when needed for specific use cases

## For TensorRT-LLM
### Optional( Read if collecting for mxfp4 kernels)
To collect performance data for mxfp4 kernels used in GPTOSS models, depending on the version of TensorRT-LLM, you might need to manually install triton-kernels. If your version is **>= 1.3.0rc2**, nothing needs to be done and you can run the collection process directly. For any version before **1.3.0rc2**. Follow the below instructions to install triton-kernels and expose them to TensorRT-LLM
1. Build and install Triton (tested with the commit below):
```bash
git clone https://github.com/triton-lang/triton.git
cd triton
# Specific commit verified with TensorRT-LLM
git checkout f3067cd3bd0c29065fa4ecdb724b6f29cbabea5f
pip install -r python/requirements.txt # build-time dependencies
pip install wheel build
python3 setup.py bdist_wheel # if this step fails due cuda related error, you might need to set CUDA_HOME following the optional step
pip install ./dist/*.whl
```
2. (Optional) You may need to set CUDA_HOME env variable to successfully build triton
```bash
export CUDA_HOME=/usr/local/cuda
export CPLUS_INCLUDE_PATH=$CUDA_HOME/include:$CPLUS_INCLUDE_PATH
export C_INCLUDE_PATH=$CUDA_HOME/include:$C_INCLUDE_PATH
python3 setup.py bdist_wheel
```
3. Expose the Triton kernels to TensorRT-LLM The kernels are not packaged in the wheel, so set the environment variable TRITON_ROOT to your Triton clone:
```bash
export TRITON_ROOT=/local/user/triton
# TensorRT-LLM expects the kernels at:
#   $TRITON_ROOT/python/triton_kernels
```
### Run collection for all available operations
```bash
python3 collect.py --backend trtllm
```
For trtllm, the whole collecting process takes about 30 gpu-hours. On 8-gpu, it takes 3-4 hours.
Please note that the whole process will report a lot of missing datapoints with errors. But it's okay. Our system is kindof robust to fair amount of missing data.
Once everything is done, you might see mutliple xxx.txt files under the same folder. Refer to src/aiconfigurator/systems/ folder to prepare the database including
how many files are needed accordingly.

## Resume Collection (Checkpoint)

Checkpoint files are always written so an interrupted run can be resumed later:

```bash
# Normal run (writes checkpoint to .collector_checkpoint/)
python3 collect.py --backend trtllm

# Resume an interrupted run (skips already-attempted tasks)
python3 collect.py --backend trtllm --resume

# Custom checkpoint directory
python3 collect.py --backend trtllm --resume --checkpoint-dir /path/to/checkpoints
```

A task is marked **done** once it is attempted (success or failure).
Only tasks that never finished are re-queued on `--resume`;
`--resume-retry-failed` additionally re-runs previously failed tasks.
Running without `--resume` always starts fresh (overwrites old checkpoint).

## For SGLang

Use the pinned stock image, `lmsysorg/sglang:v0.5.14` (or its `-cu130`
variant), for normal SGLang collection.

```bash
python3 collect.py --backend sglang
```

This collects the stock SGLang ops, including:

- GEMM operations (FP8, BF16, INT8, INT4)
- MLA (Multi-head Latent Attention) for context and generation
- MLA BMM (Batch Matrix Multiplication) operations
- MoE (Mixture of Experts) operations
- Normal attention operations

The retained `wideep_moe` op remains pinned to its separate SGLang 0.5.10 image
and is not part of the stock model plans. Request it explicitly in a separate
run. WideEP MLA is not registered because its legacy wrapper now reaches the
stock 0.5.14-only module implementation.

### DeepEP multi-node collector
For **DeepSeek V3** models with DeepEP MoE, inter-node communication data requires a separate multi-node setup:
```bash
# Follow instructions in wideep/sglang/deepep/README.md
```
See `wideep/sglang/deepep/README.md` for complete multi-node setup instructions.

# Test
Rebuild and install the new aiconfigurator. Please make sure you have your new system definition file prepared. It's src/aiconfigurator/systems/xxx.yaml

# Validate the correctness
Today, we have limited method to validate the database. You can try tools/sanity_check to validate the database a little bit. But it highly depends on your understanding
of the GPU system and kernel optimization.

# Known Issues

## NFS File Locking (Worker Deadlock)

**Symptom**: Collection stalls after a few test cases with no error messages.

**Cause**: `fcntl.flock()` doesn't work reliably on NFS. Workers deadlock when writing to shared output files.

**Solution**: Use `/tmp/` for output files, then copy results after collection.

# Support Matrix
refer to the [**support matrix CSV**](src/aiconfigurator/systems/support_matrix.csv)
