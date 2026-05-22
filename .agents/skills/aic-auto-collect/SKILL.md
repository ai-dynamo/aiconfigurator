---
name: aic-auto-collect
description: Use when running long AIC/aiconfigurator GPU perf auto-collection for a specific GPU/framework/framework_version, including draft-PR checkpoints, resumable collect_xx.py runs, framework-version/kernel preflight, collector fixes/skips, special runtime images, validation, and PR handoff.
---

# AIC Auto Collect

## Goal

You are already on a GPU node with one or more GPUs. Your job is to collect AIC perf data for one target `(GPU system, framework/backend, framework version)`, fix collector failures, recollect until the perf files are good enough, verify AIC can consume the new data, and maintain a draft PR to `ai-dynamo/aiconfigurator` as the checkpoint.

This is a long-running workflow. Be persistent. Iterate carefully. Do not hide failures by shrinking coverage or deleting hard cases unless the configuration is genuinely unsupported and documented.

## Operating Mode

Assume the user wants the complete bring-up unless they explicitly ask for a smoke test only.

- Run collectors inside the target backend runtime containers. Do not claim data was collected for a backend/version unless the collector code ran in that container on the target GPU.
- Full collection means every supported registry op for the requested backend/version has either finished with real rows or is explicitly unsupported by that runtime/GPU with evidence.
- Smoke and `--limit` runs are only gates before the full run; never present them as final data.
- Keep working through multi-hour or multi-day runs. Use stable resume checkpoints, inspect errors, patch, and retry.
- The unit of work is one draft PR per `(system, backend, backend_version)`. Use that PR as the checkpoint, not a private local branch.
- Make small sign-off commits as progress checkpoints after each `collect_xx.py`/op-family collection finishes. If a single collection runs longer than about one hour, checkpoint safe collector fixes, current perf outputs, and the failure/status summary before continuing.
- If multiple frameworks are requested, finish and checkpoint one framework/version PR first, then create a new draft PR for the next framework/version.
- Do not synthesize missing rows. Unsupported runtime/kernel paths should be filtered or documented, not filled with fake numbers.
- If the PR body says customer support is ready, verify the default AIC workflow can actually choose a valid configuration for at least the expected representative model/backend/system path.

## Required Inputs

Establish these before collecting:

- Target AIC repo and branch.
- Target backend and version: `sglang`, `vllm`, or `trtllm`. Treat "all three" as an ordered queue of separate draft PRs.
- Target runtime container image for each backend, such as:
  - `nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:<tag>`
  - `nvcr.io/nvidia/ai-dynamo/sglang-runtime:<tag>`
  - `nvcr.io/nvidia/ai-dynamo/vllm-runtime:<tag>`
- Target system name, such as `rtx_pro_6000_server`.
- Target ops: start narrow, then expand to all relevant registry ops.
- Whether to collect power columns.
- Whether the GPU type is already supported by AIC.

Discover the node:

```bash
nvidia-smi
nvidia-smi --query-gpu=name,memory.total,power.limit --format=csv
python3 - <<'PY'
import torch
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
PY
```

## Phase 1: Prepare Repo

1. Clone or update `ai-dynamo/aiconfigurator`.
2. Create a branch named like `data/<system>-<backend>-<version>` or `codex/<system>-<backend>-<version>`.
3. Immediately create a draft PR for this `(system, backend, version)` once the branch exists and the scope is known. Use it as the remote checkpoint even before all data is ready.
4. Install only lightweight local dev dependencies needed for tests. Do not install the target backend locally unless the collector is not running in a backend container.
5. Set:

```bash
export PYTHONPATH="$PWD"
export COLLECTOR_LOG_DIR="$PWD/collector_logs"
export COLLECTOR_CHECKPOINT_DIR="$PWD/collector_checkpoints"
mkdir -p "$COLLECTOR_LOG_DIR"
mkdir -p "$COLLECTOR_CHECKPOINT_DIR"
```

Local validation often works with the repo's pinned environment:

```bash
uv run --frozen ruff check <files>
uv run --frozen ruff format --check <files>
uv run --frozen pytest <tests> -q
```

If the local `uv` environment lacks framework packages such as `torch`, run framework-dependent collector smoke tests in the runtime container instead of trying to mutate the host environment.

Container pattern:

```bash
docker run --rm -it --gpus all --ipc=host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$PWD:/workspace" -w /workspace \
  <runtime-image> bash

export PYTHONPATH=/workspace
export COLLECTOR_LOG_DIR=/workspace/collector_logs/<backend>/<version>/<op>
mkdir -p "$COLLECTOR_LOG_DIR"
python collector/collect.py --backend <backend> --ops <op> --smoke
python collector/collect.py --backend <backend> --ops <op> \
  --checkpoint-dir "$COLLECTOR_CHECKPOINT_DIR/<backend>/<version>/<op>" --resume
```

Inside the container, record the real framework version:

```bash
python - <<'PY'
import importlib.metadata as m
for pkg in ("tensorrt_llm", "sglang", "vllm"):
    try:
        print(pkg, m.version(pkg))
    except Exception:
        pass
PY
```

Update the draft PR body whenever scope changes or a meaningful checkpoint lands. Include current status, what is still partial, and links or paths to failure summaries.

## Phase 2: Add Or Validate System Metadata

If the system is new or incomplete:

1. Add `src/aiconfigurator/systems/<system>.yaml`.
2. Add `<system>` to `SupportedSystems` in `src/aiconfigurator/sdk/common.py`.
3. Create `src/aiconfigurator/systems/data/<system>/`.
4. Populate YAML with conservative, documented values:
   - `gpu.mem_bw`
   - `gpu.mem_capacity`
   - tensor core FLOPS fields relevant to the architecture
   - `gpu.power`
   - `gpu.sm_version`
   - `node.num_gpus_per_node`
   - inter-node, intra-node, PCIe bandwidth
   - `misc.nccl_version`
   - `misc.other_mem`

Prefer verified values from `nvidia-smi`, node docs, or nearby existing YAML files. If a value is an estimate, leave a comment.

Run:

```bash
pytest tests/unit/sdk/test_common.py -q
```

## Phase 3: Framework Version And Kernel Preflight

Before collecting a new framework version or a new GPU SM target, check whether collector code matches the runtime. This avoids spending hours on known-bad grids.

1. Record the exact runtime package version from inside the container. Do not trust only the image tag.
2. Inspect `collector/<backend>/registry.py` for version routes and the concrete `collect_xx.py` files registered for this version.
3. Read the top of each relevant `collect_xx.py` file: docstring, `__compat__`, version notes, SM gates, and known unsupported filters.
4. Compare against the framework source for the exact version/tag when there is any sign of API or kernel churn:
   - SGLang: attention backends, DeepGEMM, FlashInfer, DeepSeek/DSV module code, MoE runner paths.
   - vLLM: attention/MLA modules, quantized GEMM paths, GDN/Mamba modules, custom all-reduce.
   - TRT-LLM: attention plugins, MLA/DSA modules, MoE, compute-scale, GDN/Mamba modules.
5. For a new SM such as SM120, check kernel architecture support before full collection: tiny smoke cases, import probes, and direct framework source guards.
6. Patch collector routing or test-case generation before full collection when the runtime cannot support a case. Use version routing and explicit SM filters rather than discovering the same failure across thousands of tasks.
7. If a special framework image exists for a model family or kernel family, try that image before declaring the op unsupported. Record its exact package version and image digest.

Commit and push preflight collector fixes before starting long full runs.

## Phase 4: Choose Collection Plan

Read the backend registry:

```bash
python3 - <<'PY'
from collector.sglang.registry import REGISTRY as SGLANG
from collector.vllm.registry import REGISTRY as VLLM
from collector.trtllm.registry import REGISTRY as TRTLLM
for name, reg in [("sglang", SGLANG), ("vllm", VLLM), ("trtllm", TRTLLM)]:
    print(name, [e.op for e in reg])
PY
```

Recommended order:

1. `gemm`
2. `moe`
3. attention / MLA ops
4. module-level ops such as DSA, GDN, MHC, WideEP
5. communication ops separately, because they may require multi-GPU or full-node ownership

For new GPUs, prioritize ops used by the target models and backend first, but do not claim full support until all expected ops for that backend/version are collected or explicitly marked unsupported.

For an all-backend bring-up, run this as separate draft PRs:

1. Finish one GPU/framework/framework_version PR.
2. Push final data and collector fixes for that framework.
3. Create the next draft PR for the next framework/version.

For each backend/version, enumerate registry ops and create a tracking table with: op, collector file, perf filename, smoke status, full status, row count, duplicate-key status, AIC sanity status, and unsupported notes. Keep this table in a local scratch file and summarize it in the PR body.

## Phase 5: Progressive Collection Loop

For each op:

```bash
python3 collector/collect.py --backend <backend> --ops <op> --smoke
python3 collector/collect.py --backend <backend> --ops <op> --shuffle --limit 20
python3 collector/collect.py --backend <backend> --ops <op> --shuffle --limit 100
python3 collector/collect.py --backend <backend> --ops <op> \
  --checkpoint-dir "$COLLECTOR_CHECKPOINT_DIR/<backend>/<version>/<op>" --resume
```

Use a separate log directory per op when iterating:

```bash
export COLLECTOR_LOG_DIR="$PWD/collector_logs/<backend>/<version>/<op>"
export COLLECTOR_CHECKPOINT_DIR="$PWD/collector_checkpoints"
mkdir -p "$COLLECTOR_LOG_DIR"
mkdir -p "$COLLECTOR_CHECKPOINT_DIR/<backend>/<version>/<op>"
```

If the process times out or crashes after partial progress, rerun with the same `--checkpoint-dir` and `--resume`. Keep checkpoints until the op is accepted. If the process has been running for about an hour, checkpoint safe progress:

- Commit collector fixes and tests.
- Copy currently produced perf files only if they are clearly labeled or known to be append-safe for replacement by a later full run.
- Push the draft PR branch.
- Update the PR body or a tracked status note with current op, completed cases, row counts, and active failures.

Full-collection acceptance for an op:

- The final collector run has zero unclassified errors, or every remaining skipped case is intentionally filtered before execution.
- The output perf file was copied from the full output directory, not from an earlier smoke/limited run.
- Row counts are plausible compared with nearby systems or the generated case count.
- Duplicate shape keys are removed only when they are true repeated measurements for the same key; keep the latest or best-defined row consistently and document the policy.
- The op is not required by AIC's default query path, or if it is required, representative AIC CLI/chart generation can consume it.

For very long runs, commit and push safe progress periodically:

```bash
git status --short
git add collector tests src/aiconfigurator/systems
git commit --signoff -m "<backend>: <op family> progress"
git push
```

Use sign-off commits. If DCO fails because a commit lacks `Signed-off-by`, amend only the latest commit if appropriate:

```bash
git commit --amend --no-edit --signoff
git push --force-with-lease
```

## Phase 6: Fix Collector Errors

When a run fails, inspect:

- `collection_summary_<backend>.json`
- `errors_*.json`
- worker logs in `COLLECTOR_LOG_DIR`
- traceback module and task parameters

Maintain a failure ledger for the active PR. For each failure group record: op, collector file, runtime image/version, SM, task params, exact error, classification, action taken, and whether an upstream issue was filed.

Classify each error group:

- **collector_bug**: import/API/mock object/signature/test generation issue.
- **unsupported_config**: generated case is invalid for this SM/backend/kernel.
- **framework_bug**: backend kernel crashes or rejects a valid production-like case.
- **resource_issue**: OOM, shared memory, graph capture, worker restart, timeout.
- **transient**: rare cache/race/infra issue that passes on retry.

Fix policy:

- For API/import/signature errors:
  1. Read `collector/<backend>/registry.py` to map the failing op to its collector module.
  2. Read the failing `collector/<backend>/collect_*.py`.
  3. Search the framework source checkout for the missing class, function, import path, or attribute.
  4. Verify the real API in source. Do not trust Python's "Did you mean" suggestions blindly.
  5. Patch the collector with a minimal compatibility-preserving change.
- For API changes, use version routing and `__compat__` where needed.
- For GPU capability gaps, filter test cases before execution using `get_sm_version()`.
- For Blackwell/SM100+ features, gate FP4/NVFP4/FP8 paths carefully.
- For dimension constraints, filter on per-rank dimensions such as `inter_size // tp`.
- For CUDA-fatal errors, prefer worker restart and skip only the exact invalid region.
- Preserve data quality. Do not reduce benchmark repetitions or broad test dimensions just to pass.
- Keep older backend versions working.
- Keep shared test-case generators deterministic when unit tests depend on them. Put runtime availability probes in backend collector wrappers when the skip is caused by a partially installed framework package.
- Guard `importlib.util.find_spec("a.b.c")` with `try/except ModuleNotFoundError`; partial packages can make `find_spec` raise instead of return `None`.
- If a collector route only exists for future framework versions, use `VersionRoute` in `collector/<backend>/registry.py` and add a wrapper module with an explicit `__compat__`.
- If a collector helper silently returns `None` after a dry-run or subprocess failure, make it raise or record the skip explicitly so failed coverage is visible.
- If a valid production-like case fails in the framework kernel, file or prepare an upstream bug for SGLang, vLLM, or TRT-LLM with exact image/version, SM, minimal repro, and traceback. Link it in the PR or failure ledger.
- If the case is not production-like or the runtime explicitly does not support that SM/path, skip it in `collect_xx.py` before execution with a clear comment and a narrow predicate.

After a fix:

```bash
pytest tests/unit/collector -q
python3 collector/collect.py --backend <backend> --ops <op> --smoke
python3 collector/collect.py --backend <backend> --ops <op> --shuffle --limit 100
python3 collector/collect.py --backend <backend> --ops <op> \
  --checkpoint-dir "$COLLECTOR_CHECKPOINT_DIR/<backend>/<version>/<op>" --resume
```

Commit small fixes:

```bash
git add collector tests
git commit --signoff -m "fix <backend> <op> collector for <system>"
```

### Backend-Specific Failure Patterns

Use these as starting hypotheses, then verify against the actual runtime source and logs.

TRT-LLM:

- Compute-scale latency may legitimately be zero when `max(0, dynamic_quantize - static_quantize)` clamps a dynamic path that measured faster than the static baseline. Verify with remeasurement; do not invent positive latency.
- Some GDN collectors require newer TRT-LLM than the runtime image provides. Route by framework version instead of registering an unusable op.
- DSA or sparse MLA modules may be unsupported for a new SM/runtime combination. Filter unsupported cases before execution and document the runtime limitation.
- For Blackwell/SM120 attention and FP8/FP4 paths, check per-rank dimensions, kv dtype, and kernel architecture support before collecting.

SGLang:

- DeepGEMM, FlashInfer, DSV4, MHC, and MLA paths can be present in source but unavailable for the active SM or installed package set.
- MLA prefill/generation may have hard-coded head/group constraints on new architectures. Restrict to verified safe shapes rather than allowing fatal CUDA crashes.
- If `sglang` is partially installed in a unit-test image, `sglang.srt...` probes can raise. Runtime skip checks belong in `collector/sglang/collect_*.py` wrappers, not necessarily in `collector/common_test_cases.py`.
- For DSV4/MHC, distinguish "case grid exists" from "runtime module exists"; unit tests may assert the former while the collector should enforce the latter.

vLLM:

- Newer ops such as GDN may not exist in older vLLM runtime images. Add version routing, for example route GDN only for `vllm>=0.17.0` if `0.16.0` lacks the collector API.
- FP8 block GEMM and quantized GEMM paths can have high memory use or graph/eager timing differences. Prefer the runtime-supported timing mode and reduce memory pressure without reducing intended coverage.
- FP8 KV-cache attention or MLA module combinations may not be supported by the installed runtime. Filter unsupported kv dtype/module combinations early.
- Cap generation module cases that exceed runtime limits such as very large `batch * sequence` regions, and document the cap.
- When copying final vLLM data, scan for duplicate keys. Some repeated collection paths can produce duplicate shape rows.

## Phase 7: Perf File Quality Gates

Accept a perf file only when:

- It exists under `src/aiconfigurator/systems/data/<system>/<backend>/<version>/`.
- It is not empty and has the expected CSV header.
- Rows contain the expected framework, version, and device name.
- Latency values are finite and plausible. They should usually be positive; allow zero only for documented modeled-overhead files such as compute-scale where the collector intentionally clamps negative overhead to zero.
- Power values, when collected, are positive and below/near the configured power limit.
- There are no unexplained duplicate rows for identical keys.
- Coverage is comparable to the nearest existing system/backend/version.
- Missing cases are explained by documented unsupported config filters.
- Re-running sample does not produce large unexplained variance.

Useful checks:

```bash
find src/aiconfigurator/systems/data/<system>/<backend>/<version> -maxdepth 1 -type f -name '*_perf.txt' -print
python3 - <<'PY'
from pathlib import Path
import csv, math
root = Path("src/aiconfigurator/systems/data/<system>/<backend>/<version>")
for path in sorted(root.glob("*_perf.txt")):
    with path.open() as f:
        rows = list(csv.DictReader(f))
    bad = [r for r in rows if "latency" in r and (not r["latency"] or not math.isfinite(float(r["latency"])) or float(r["latency"]) <= 0)]
    print(path.name, "rows", len(rows), "bad_latency", len(bad))
PY
```

Compare row counts and latency ranges with nearby systems, such as H100/H200 for Hopper or B200/B300 for Blackwell.

Run a duplicate-key scan using the loader's expected key columns when possible. If no helper exists, group rows by all non-metric columns and report duplicates per file. Do not blindly dedupe rows before understanding whether repeated dimensions differ by dtype, backend, tensor parallelism, or model field.

## Phase 8: Verify AIC Consumption

Run loader and SDK tests:

```bash
pytest tests/unit/sdk/test_common.py -q
pytest tests/unit/sdk/database -q
pytest tests/unit/collector -q
```

Instantiate the database:

```bash
python3 - <<'PY'
from aiconfigurator.sdk.perf_database import get_database
db = get_database("<system>", "<backend>", "<version>")
print(db is not None)
print(db.system_spec["gpu"])
PY
```

Run representative CLI queries for models expected to use this backend/system:

```bash
aiconfigurator cli default --model <model> --total-gpus <n> --system <system> --backend <backend>
aiconfigurator cli generate --model-path <model> --total-gpus <n> --system <system> --backend <backend>
```

If AIC raises `PerfDataNotAvailableError`, either collect the missing op or document that the model/backend mode is not supported. Do not claim support for a model path that still hits missing data.

Run sanity chart generation before declaring the PR ready:

```bash
rm -rf sanity_<system> && mkdir -p sanity_<system>
uv run --frozen python tools/sanity_check/create_charts.py \
  --base-ref origin/main \
  --head-ref HEAD \
  --output-dir sanity_<system> \
  --output-md-file sanity_<system>/comment.md
```

Read the generated report, not just the exit code. Fix hard failures where possible:

- Missing data for default workflow shapes usually means collection coverage is incomplete.
- Empty chart grids can be acceptable only when the runtime has no valid silicon data for that op/shape family; convert those to explicit skips with a clear reason.
- If both aggregated and disaggregated configs have no valid parallel configuration, the backend/version is registered but not usable. Collect the missing required shapes or do not claim default workflow support.
- CLI smoke in the report must pass for each backend/version claimed ready.

## Phase 9: Support Matrix And Documentation

If the new GPU should become user-visible:

1. Update system support metadata.
2. Update support matrix outputs if required by the repo workflow.
3. Add or update tests for new SM filters or version routes.
4. Document known gaps in the PR body.

## Phase 10: PR Preparation

Keep commits reviewable:

1. `add <system> system spec`
2. `fix <backend> collectors for <system>`
3. `add <system> <backend> <version> perf data`
4. `test <system> data loading`

Before marking the draft PR ready:

```bash
git status --short
git diff --stat main...HEAD
pytest tests/unit/collector -q
pytest tests/unit/sdk/test_common.py -q
```

PR body checklist:

- A concrete "Release at a Glance"; avoid generic claims like "metrics below show improvements" unless the tables explain those improvements.
- Customer impact: what is now selectable/usable in AIC, and for which default model/backend/system path.
- GPU node identity and `nvidia-smi` summary.
- CUDA driver/container/backend image for each backend.
- Backend version detected at runtime, not assumed from the image tag.
- System YAML changes.
- Ops collected and perf files added.
- Row counts by backend/version and perf file count.
- Collector errors fixed.
- Unsupported/skipped configs with reasons and whether they are runtime-version limits, SM limits, or collector gaps.
- Validation commands and results.
- Remaining risks.

If GitHub CLI PR editing fails because of deprecated project-card GraphQL fields, update the PR body through the REST API:

```bash
gh api repos/<owner>/<repo>/pulls/<number> -X PATCH -f body="$(cat pr_body.md)"
```

Check CI after every push:

```bash
gh pr checks <number>
gh run view <run-id> --log-failed
```

Ruff CI runs both `ruff check` and `ruff format --check`; local targeted `ruff check` alone is not enough.

If the draft PR does not exist yet, open it before more long-running work:

```bash
gh repo fork ai-dynamo/aiconfigurator --clone=false
git push -u <your-fork-remote> HEAD
gh pr create --repo ai-dynamo/aiconfigurator --head <your-user>:<branch> --title "<title>" --body-file <body-file>
```

## Stop And Escalate

Escalate instead of looping forever when:

- The framework kernel crashes consistently on valid production-like configs.
- The backend container cannot run on the node.
- The GPU platform or driver is unstable.
- The system spec values are unknown and materially affect AIC decisions.
- Data quality is suspect and cannot be explained.
- AIC requires broad SDK/model changes beyond collector/data bring-up.

In the final report, separate collected data, collector fixes, known gaps, and validation evidence.
