#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render a Slurm job for DSv4 MegaMoE collection.

The renderer only writes an sbatch script. Submission, monitoring, result
download, and cleanup stay in ``run_slurm_full_collection.sh`` so resource
lifetime is explicit.
"""

from __future__ import annotations

import argparse
import shlex

DEFAULT_GPUS_PER_NODE = {
    "B200": 8,
    "B200_SXM": 8,
    "GB200": 4,
    "B300": 8,
    "B300_SXM": 8,
    "GB300": 4,
}

DEFAULT_IMAGE = "lmsysorg/sglang-staging:deepseek-v4-grace-blackwell-dev"


def _default_gpus_per_node(system_name: str) -> int:
    try:
        return DEFAULT_GPUS_PER_NODE[system_name.upper()]
    except KeyError as exc:
        raise SystemExit(f"--gpus-per-node is required for unknown system {system_name}") from exc


def _q(value: object) -> str:
    return shlex.quote(str(value))


def _export(name: str, value: object) -> str:
    return f"export {name}={_q(value)}"


def _parse_extra_env(values: list[str]) -> list[tuple[str, str]]:
    envs = []
    for item in values:
        if "=" not in item:
            raise SystemExit(f"--env must be KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        if not key:
            raise SystemExit(f"--env key must not be empty, got {item!r}")
        envs.append((key, value))
    return envs


def render(args: argparse.Namespace) -> str:
    gpus_per_node = args.gpus_per_node or _default_gpus_per_node(args.system_name)
    if args.ep_size % gpus_per_node != 0:
        raise SystemExit("--ep-size must be divisible by --gpus-per-node")
    nnodes = args.ep_size // gpus_per_node

    exports = [
        _export("REMOTE_WORKDIR", args.remote_workdir),
        _export("OUTPUT_PATH", args.output_path),
        _export("PERF_FILE", args.perf_file),
        _export("SYSTEM_NAME", args.system_name),
        _export("MODEL_CONFIG", args.model_config),
        _export("GPUS_PER_NODE", gpus_per_node),
        _export("EP_SIZE", args.ep_size),
        _export("NNODES", nnodes),
        _export("PHASES", args.phase),
        _export("PREFILL_TOKENS", args.prefill_tokens),
        _export("DECODE_TOKENS", args.decode_tokens),
        _export("DISTRIBUTIONS", args.distributions),
        _export("SOURCE_POLICY", args.source_policy),
        _export("ROUTING_SEED", args.routing_seed),
        _export("ROUTING_SEEDS", args.routing_seeds),
        _export("PRE_DISPATCH", args.pre_dispatch),
        _export("INCLUDE_ROUTED_SCALE", args.include_routed_scale),
        _export("RENORMALIZE_TOPK_WEIGHTS", args.renormalize_topk_weights),
        _export("NUM_WARMUP", args.num_warmup),
        _export("NUM_ITERATIONS", args.num_iterations),
        _export("NUM_MAX_TOKENS_PER_RANK", args.num_max_tokens_per_rank),
        _export("CAP_POLICY", args.cap_policy),
        _export("CONTAINER_IMAGE", args.container_image),
        _export("CONTAINER_MOUNTS", args.container_mounts),
        _export("CONTAINER_WRITABLE", 1 if args.container_writable else 0),
    ]
    for key, value in _parse_extra_env(args.env):
        exports.append(_export(key, value))

    exclusive = "#SBATCH --exclusive\n" if args.exclusive else ""
    export_block = "\n".join(exports)

    return f"""#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#SBATCH --job-name={args.job_name}
#SBATCH --account={args.account}
#SBATCH --partition={args.partition}
#SBATCH --nodes={nnodes}
#SBATCH --ntasks={args.ep_size}
#SBATCH --ntasks-per-node={gpus_per_node}
#SBATCH --time={args.time_limit}
{exclusive}#SBATCH --output={args.log_dir}/%x-%j.out
#SBATCH --error={args.log_dir}/%x-%j.err

set -euo pipefail

{export_block}

mkdir -p "${{OUTPUT_PATH}}"

MASTER_ADDR="$(scontrol show hostnames "${{SLURM_JOB_NODELIST}}" | head -n 1)"
MASTER_PORT="${{MASTER_PORT:-$((29500 + SLURM_JOB_ID % 10000))}}"
export MASTER_ADDR MASTER_PORT

export NCCL_DEBUG="${{NCCL_DEBUG:-WARN}}"
export NCCL_GRAPH_MIXING_SUPPORT="${{NCCL_GRAPH_MIXING_SUPPORT:-0}}"
export NCCL_NVLS_ENABLE="${{NCCL_NVLS_ENABLE:-1}}"
if [[ "${{SYSTEM_NAME^^}}" == "GB200" && "${{NNODES}}" -gt 1 ]]; then
  export NCCL_MNNVL_ENABLE="${{NCCL_MNNVL_ENABLE:-1}}"
  export NCCL_CUMEM_ENABLE="${{NCCL_CUMEM_ENABLE:-1}}"
fi
export SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE="${{SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE:-1}}"
export SGLANG_OPT_FIX_HASH_MEGA_MOE="${{SGLANG_OPT_FIX_HASH_MEGA_MOE:-1}}"
export SGLANG_OPT_FIX_MEGA_MOE_MEMORY="${{SGLANG_OPT_FIX_MEGA_MOE_MEMORY:-1}}"
export SGLANG_OPT_FIX_NEXTN_MEGA_MOE="${{SGLANG_OPT_FIX_NEXTN_MEGA_MOE:-1}}"
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK="${{NUM_MAX_TOKENS_PER_RANK}}"
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK="${{SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-0}}"
export AIC_SYSTEM_NAME="${{SYSTEM_NAME}}"

read -r -d '' AIC_SLURM_WORKLOAD <<'AIC_WORKLOAD' || true
set -euo pipefail
_write_rank0_env() {{
  local allowed_env_regex
  allowed_env_regex='^(AIC_|CUDA_|NCCL_|SGLANG_|TORCH_|SLURM_JOB_ID=|SLURM_JOB_NAME=|SLURM_NODELIST='
  allowed_env_regex+='|SLURM_NNODES=|SLURM_NTASKS=|SLURM_PROCID=|SLURM_LOCALID=|RANK=|WORLD_SIZE='
  allowed_env_regex+='|LOCAL_RANK=|NNODES=|MASTER_ADDR=|MASTER_PORT=|MODEL_CONFIG=|SYSTEM_NAME='
  allowed_env_regex+='|GPUS_PER_NODE=|EP_SIZE=|OUTPUT_PATH=|PERF_FILE=|PHASES=|PREFILL_TOKENS='
  allowed_env_regex+='|DECODE_TOKENS=|DISTRIBUTIONS=|SOURCE_POLICY=|PRE_DISPATCH='
  allowed_env_regex+='|NUM_MAX_TOKENS_PER_RANK=|CAP_POLICY=)'
  {{
    env | sort \\
      | grep -E "${{allowed_env_regex}}" \\
      | grep -Evi '(TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH|PRIVATE|SSH|KEY)' \\
      || true
  }} >"${{OUTPUT_PATH}}/rank0_env.txt"
}}
cd "${{REMOTE_WORKDIR}}"
SGLANG_PYTHONPATHS=()
for candidate in /workspace/sglang/python /sgl-workspace/sglang/python; do
  if [[ -d "${{candidate}}" ]]; then
    SGLANG_PYTHONPATHS+=("${{candidate}}")
  fi
done
if ((${{#SGLANG_PYTHONPATHS[@]}})); then
  SGLANG_PYTHONPATH="$(IFS=:; echo "${{SGLANG_PYTHONPATHS[*]}}")"
  export PYTHONPATH="${{REMOTE_WORKDIR}}:${{SGLANG_PYTHONPATH}}:${{PYTHONPATH:-}}"
else
  export PYTHONPATH="${{REMOTE_WORKDIR}}:${{PYTHONPATH:-}}"
fi
mkdir -p "${{OUTPUT_PATH}}"
export RANK="${{SLURM_PROCID:?}}"
export WORLD_SIZE="${{SLURM_NTASKS:?}}"
export LOCAL_RANK="${{SLURM_LOCALID:-0}}"
SGLANG_VERSION="$(python3 - <<'PY'
from importlib.metadata import PackageNotFoundError, version

try:
    print(version("sglang"))
except PackageNotFoundError:
    print("unknown")
PY
)"
export SGLANG_VERSION
if [[ "${{SLURM_PROCID:-0}}" == "0" ]]; then
  printf '%s\\n' "${{SGLANG_VERSION}}" >"${{OUTPUT_PATH}}/sglang_version.txt"
  _write_rank0_env
fi
python3 collector/sglang/collect_dsv4_megamoe.py \\
  --model-config "${{MODEL_CONFIG}}" \\
  --system-name "${{SYSTEM_NAME}}" \\
  --gpus-per-node "${{GPUS_PER_NODE}}" \\
  --phases "${{PHASES}}" \\
  --prefill-tokens "${{PREFILL_TOKENS}}" \\
  --decode-tokens "${{DECODE_TOKENS}}" \\
  --distributions "${{DISTRIBUTIONS}}" \\
  --source-policy "${{SOURCE_POLICY}}" \\
  --routing-seed "${{ROUTING_SEED}}" \\
  --routing-seeds "${{ROUTING_SEEDS}}" \\
  --pre-dispatch "${{PRE_DISPATCH}}" \\
  --include-routed-scale "${{INCLUDE_ROUTED_SCALE}}" \\
  --renormalize-topk-weights "${{RENORMALIZE_TOPK_WEIGHTS}}" \\
  --num-warmup "${{NUM_WARMUP}}" \\
  --num-iterations "${{NUM_ITERATIONS}}" \\
  --num-max-tokens-per-rank "${{NUM_MAX_TOKENS_PER_RANK}}" \\
  --cap-policy "${{CAP_POLICY}}" \\
  --output-path "${{OUTPUT_PATH}}" \\
  --perf-file "${{PERF_FILE}}" \\
  --sglang-version "${{SGLANG_VERSION}}"
AIC_WORKLOAD

container_args=(
  --container-image="${{CONTAINER_IMAGE}}"
  --container-mounts="${{CONTAINER_MOUNTS}}"
  --container-workdir="${{REMOTE_WORKDIR}}"
)
if [[ "${{CONTAINER_WRITABLE}}" == "1" ]]; then
  container_args+=(--container-writable)
fi

srun "${{container_args[@]}}" bash -lc "${{AIC_SLURM_WORKLOAD}}"
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--account", default="coreai_tritoninference_triton3")
    parser.add_argument("--partition", default="gb300")
    parser.add_argument("--time-limit", default="02:00:00")
    parser.add_argument("--system-name", default="gb300")
    parser.add_argument("--ep-size", type=int, required=True)
    parser.add_argument("--gpus-per-node", type=int, default=None)
    parser.add_argument("--phase", choices=["context", "generation"], required=True)
    parser.add_argument("--remote-workdir", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--container-image", default=DEFAULT_IMAGE)
    parser.add_argument("--container-mounts", default="/lustre:/lustre")
    parser.add_argument("--container-writable", action="store_true")
    parser.add_argument("--no-exclusive", dest="exclusive", action="store_false")
    parser.set_defaults(exclusive=True)
    parser.add_argument("--model-config", default="dsv4_pro")
    parser.add_argument("--perf-file", default="dsv4_megamoe_module_perf.txt")
    parser.add_argument("--prefill-tokens", default="1024,2048,4096,8192,16384,32768")
    parser.add_argument("--decode-tokens", default="1,2,4,8,16,32,64,128,256,512")
    parser.add_argument(
        "--distributions",
        default="balanced,power_law_1.01,power_law_1.2,power_law_sampled_1.9",
    )
    parser.add_argument("--source-policy", choices=["random"], default="random")
    parser.add_argument("--routing-seed", type=int, default=0)
    parser.add_argument("--routing-seeds", default="")
    parser.add_argument("--pre-dispatch", default="sglang_jit")
    parser.add_argument("--include-routed-scale", type=int, default=1)
    parser.add_argument("--renormalize-topk-weights", type=int, default=1)
    parser.add_argument("--num-warmup", type=int, default=5)
    parser.add_argument("--num-iterations", type=int, default=20)
    parser.add_argument("--num-max-tokens-per-rank", type=int, required=True)
    parser.add_argument("--cap-policy", choices=["fixed", "case_tokens"], default="case_tokens")
    parser.add_argument("--env", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    print(render(parse_args()), end="")


if __name__ == "__main__":
    main()
