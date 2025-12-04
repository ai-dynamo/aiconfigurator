#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# ------------------------------------------------------------
# Usage:
#   ./launch_eval.sh /path/to/config.env
#
# The config file is a simple VAR=VALUE list.
# ------------------------------------------------------------

# ---------- utils ----------
log()  { echo "[INFO ] $*" >&2; }
warn() { echo "[WARN ] $*" >&2; }
err()  { echo "[ERROR] $*" >&2; exit 1; }

have() { command -v "$1" >/dev/null 2>&1; }

require() {
  for c in "$@"; do
    have "$c" || err "Missing command: $c"
  done
}

docker_compose() {
  if docker compose version >/dev/null 2>&1; then
    docker compose "$@"
  elif have docker-compose; then
    docker-compose "$@"
  else
    return 127
  fi
}

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )"
SRC_ROOT_HOST="$(realpath "$SCRIPT_DIR/../..")"     # project root to mount into container
AICONF_SRC_CONTAINER="/opt/aiconfigurator-src"      # mount point inside container

# ---------- args ----------
CONFIG_FILE="${1:-}"
[[ -n "$CONFIG_FILE" && -f "$CONFIG_FILE" ]] || err "Config file missing: $CONFIG_FILE"
# shellcheck source=/dev/null
source "$CONFIG_FILE"

# ---------- defaults ----------
: "${ENABLE_MODEL_DOWNLOAD:=true}"
: "${IN_CONTAINER:=false}"

: "${MODEL_LOCAL_DIR:?MODEL_LOCAL_DIR is required}"
: "${MODEL_HF_REPO:?MODEL_HF_REPO is required}"

: "${SYSTEM:?SYSTEM is required}"
: "${MODEL:?MODEL is required}"                # e.g. QWEN3_32B
: "${VERSION:?VERSION is required}"
: "${ISL:?ISL is required}"
: "${OSL:?OSL is required}"
: "${TTFT:?TTFT is required}"
: "${TPOT:?TPOT is required}"
: "${TOTAL_GPUS:?TOTAL_GPUS is required}"
: "${HEAD_NODE_IP:?HEAD_NODE_IP is required}"
: "${MODE:?MODE is required}"   # disagg | agg


: "${PREFILL_FREE_GPU_MEM_FRAC:=0.9}"
: "${FREE_GPU_MEM_FRAC:=0.7}"
: "${DECODE_FREE_GPU_MEM_FRAC:=0.6}"

: "${SERVED_MODEL_NAME:?SERVED_MODEL_NAME is required}"

: "${PORT:=8000}"
: "${VENV_PATH:=/workspace/aic}"

# Container/image settings
: "${DYNAMO_IMAGE:=}"
: "${TRTLLM_PIP:=}"                               # e.g. tensorrt-llm==1.0.0rc4
: "${DYNAMO_DIR:=}"                               # optional pre-existing repo dir
: "${DYNAMO_BRANCH:=main}"
: "${DYNAMO_GIT:=https://github.com/ai-dynamo/dynamo}"

: "${CONTAINER_NAME:=dynamo-single-node}"
: "${BENCHMARK_CONCURRENCY:=}"   # empty or 'auto' -> auto mode


# Paths inside container
CONTAINER_MODEL_PATH="${MODEL_PATH:-/workspace/model_hub/$(basename "$MODEL_LOCAL_DIR")}"
CONTAINER_SAVE_DIR="${SAVE_DIR:-/workspace/aiconf_save}"

# Host-side save dir (used when IN_CONTAINER=false, mounted to container)
SAVE_DIR_HOST="${SAVE_DIR_HOST:-$PWD/aiconf_save}"

# HF token is optional; forward if present
: "${HF_TOKEN:=}"

ensure_model_local() {
  if [[ "${ENABLE_MODEL_DOWNLOAD,,}" != "true" ]]; then
    log "Model download disabled by config."
    return 0
  fi
  if [[ -d "$MODEL_LOCAL_DIR" ]] && find "$MODEL_LOCAL_DIR" -mindepth 1 -print -quit | grep -q .; then
    log "Model already present at ${MODEL_LOCAL_DIR}"
    return 0
  fi

  log "Downloading model to ${MODEL_LOCAL_DIR} (repo: ${MODEL_HF_REPO})"
  mkdir -p "$MODEL_LOCAL_DIR"

  if have huggingface-cli; then
    if [[ -n "$HF_TOKEN" ]]; then export HF_TOKEN; fi
    huggingface-cli download \
      --repo-type model "$MODEL_HF_REPO" \
      --local-dir "$MODEL_LOCAL_DIR" \
      --local-dir-use-symlinks False
  elif have git; then
    require git
    have git-lfs || warn "git-lfs not found; trying plain git (may be slow or incomplete)"
    GIT_URL="https://huggingface.co/${MODEL_HF_REPO}"
    git lfs install || true
    if [[ ! -d "$MODEL_LOCAL_DIR/.git" ]]; then
      git clone --depth 1 "$GIT_URL" "$MODEL_LOCAL_DIR" || git clone "$GIT_URL" "$MODEL_LOCAL_DIR"
    fi
  else
    err "No downloader found. Install huggingface-cli or git."
  fi

  if ! find "$MODEL_LOCAL_DIR" -mindepth 1 -print -quit | grep -q .; then
    err "Model directory is empty: ${MODEL_LOCAL_DIR}"
  fi
}

ensure_dynamo_repo() {
  if [[ -n "$DYNAMO_DIR" && -d "$DYNAMO_DIR/.git" ]]; then
    log "Using existing dynamo repo: $DYNAMO_DIR"
    (cd "$DYNAMO_DIR" && git fetch --all -p && git checkout "$DYNAMO_BRANCH" && git pull --ff-only || true)
    return 0
  fi
  DYNAMO_DIR="${DYNAMO_DIR:-$PWD/dynamo}"
  if [[ -d "$DYNAMO_DIR/.git" ]]; then
    log "Using existing dynamo repo: $DYNAMO_DIR"
    (cd "$DYNAMO_DIR" && git fetch --all -p && git checkout "$DYNAMO_BRANCH" && git pull --ff-only || true)
  else
    require git
    log "Cloning dynamo repo -> $DYNAMO_DIR (branch: $DYNAMO_BRANCH)"
    git clone --branch "$DYNAMO_BRANCH" --depth 1 "$DYNAMO_GIT" "$DYNAMO_DIR"
  fi
}

ensure_compose_stack() {
  require docker
  ensure_dynamo_repo

  local compose_file="$DYNAMO_DIR/deploy/docker-compose.yml"
  if [[ ! -f "$compose_file" ]]; then
    warn "Compose file not found: $compose_file (skip)"
    return 0
  fi
  log "Starting compose stack: $compose_file"
  (
    set +e
    cd "$DYNAMO_DIR" || exit 0
    docker_compose -f "$compose_file" up -d
    rc=$?
    if [[ $rc -ne 0 ]]; then
      warn "Compose up failed with rc=$rc; continue anyway."
    else
      log "Compose stack is up."
    fi
  )
  return 0
}

ensure_image() {
  require docker

  local user_set_image="false"
  if [[ -n "${DYNAMO_IMAGE:-}" ]]; then
    user_set_image="true"
  fi

  if [[ -z "$DYNAMO_IMAGE" ]]; then
    local tag_ver="local"
    if [[ -n "$TRTLLM_PIP" ]]; then
      tag_ver="${TRTLLM_PIP#*=}"
    fi
    DYNAMO_IMAGE="dynamo:0.4.0-trtllm-${tag_ver}"
    log "DYNAMO_IMAGE not set. Using ${DYNAMO_IMAGE}"
  fi

  if docker image inspect "$DYNAMO_IMAGE" >/dev/null 2>&1; then
    log "Image exists: $DYNAMO_IMAGE"
    return 0
  fi

  if [[ "$user_set_image" == "true" ]]; then
    log "Attempting to pull image: $DYNAMO_IMAGE"
    if docker pull "$DYNAMO_IMAGE" >/dev/null 2>&1; then
      log "Pulled image: $DYNAMO_IMAGE"
      return 0
    else
      warn "Pull failed for $DYNAMO_IMAGE; will try local build if possible."
    fi
  fi

  [[ -n "$TRTLLM_PIP" ]] || err "Cannot build image: TRTLLM_PIP is required (e.g., tensorrt-llm==1.0.0rc4)"
  ensure_dynamo_repo

  local framework="TRTLLM"
  if [[ -n "$DYNAMO_BRANCH" && "$DYNAMO_BRANCH" =~ 0\.([0-9]+)\.([0-9]+)$ ]]; then
    local x="${BASH_REMATCH[1]}"
    local x_dec=$((10#$x))
    if (( x_dec <= 4 )); then
      framework="tensorrtllm"
    fi
  fi

  log "Building image: $DYNAMO_IMAGE (framework: $framework, branch: ${DYNAMO_BRANCH:-<unset>})"
  (
    cd "$DYNAMO_DIR"
    ./container/build.sh \
      --framework "$framework" \
      --tensorrtllm-pip-wheel "$TRTLLM_PIP" \
      --tag "$DYNAMO_IMAGE"
  )
  log "Image built: $DYNAMO_IMAGE"
}

compose_eval_cmd() {
  local model_path_arg
  if [[ "${IN_CONTAINER,,}" == "true" ]]; then
    if [[ -n "${MODEL_PATH:-}" ]]; then
      model_path_arg="$MODEL_PATH"
    else
      model_path_arg="$MODEL_LOCAL_DIR"
    fi
  else
    model_path_arg="$CONTAINER_MODEL_PATH"
  fi

  local BC_ARG=""
  if [[ -n "${BENCHMARK_CONCURRENCY:-}" && "${BENCHMARK_CONCURRENCY,,}" != "auto" ]]; then
    BC_ARG="--benchmark-concurrency ${BENCHMARK_CONCURRENCY}"
  fi

  EVAL_CMD=$(cat <<-EOF
aiconfigurator eval \
  --service-mode "$MODE" \
  ${BC_ARG} \
  --venv-dir "$VENV_PATH" \
  default \
  --model "$MODEL" \
  --total_gpus "$TOTAL_GPUS" \
  --head_node_ip "$HEAD_NODE_IP" \
  --port "$PORT" \
  --system "$SYSTEM" \
  --backend_version "$VERSION" \
  --generated_config_version "$GENERATED_CONFIG_VERSION" \
  --generator-set ServiceConfig.model_path="$model_path_arg" \
  --generator-set ServiceConfig.served_model_name="$SERVED_MODEL_NAME" \
  --generator-set Workers.prefill.kv_cache_free_gpu_memory_fraction="$PREFILL_FREE_GPU_MEM_FRAC" \
  --generator-set Workers.decode.kv_cache_free_gpu_memory_fraction="$DECODE_FREE_GPU_MEM_FRAC" \
  --generator-set Workers.agg.kv_cache_free_gpu_memory_fraction="$FREE_GPU_MEM_FRAC" \
  --isl "$ISL" --osl "$OSL" --ttft "$TTFT" --tpot "$TPOT" \
  --save_dir "$CONTAINER_SAVE_DIR"
EOF
)
}

run_inside_container() {
  log "Running inside container (IN_CONTAINER=true)"
  ensure_model_local
  compose_eval_cmd

  # ensure 'aiconfigurator' command exists; if not, install from repo
  local SRC_ROOT_INNER
  SRC_ROOT_INNER="$(realpath "$SCRIPT_DIR/../..")"
  if ! command -v aiconfigurator >/dev/null 2>&1; then
    log "Installing aiconfigurator from source: $SRC_ROOT_INNER"
    python3 -m ensurepip --upgrade >/dev/null 2>&1 || true
    (cd "$SRC_ROOT_INNER" && python3 -m pip install -e . --break-system-packages)

    # Prepare uv virtualenv for aiperf
    uv venv "$VENV_PATH"
    source "$VENV_PATH/bin/activate"
    uv pip install aiperf
    deactivate
  fi

  log "Eval command:\n$EVAL_CMD"
  eval "$EVAL_CMD"
}

run_with_docker() {
  require docker
  ensure_model_local

  ensure_compose_stack

  ensure_image

  mkdir -p "$SAVE_DIR_HOST"
  compose_eval_cmd

  local RUN_DIR_HOST="$SAVE_DIR_HOST/.automation"
  local RUN_SCRIPT_HOST="$RUN_DIR_HOST/run_in_container.sh"
  local RUN_SCRIPT_CONT="$CONTAINER_SAVE_DIR/.automation/run_in_container.sh"
  mkdir -p "$RUN_DIR_HOST"

  cat > "$RUN_SCRIPT_HOST" <<'BASH'
#!/usr/bin/env bash
set -euo pipefail

# Bootstrap: ensure 'aiconfigurator' exists
if ! command -v aiconfigurator >/dev/null 2>&1; then
  echo "[BOOT] installing aiconfigurator from $AICONF_SRC"
  python3 -m ensurepip --upgrade >/dev/null 2>&1 || true
  cd "$AICONF_SRC"
  python3 -m pip install -e . --break-system-packages

  # Prepare uv virtualenv for aiperf
  uv venv "$VENV_PATH"
  source "$VENV_PATH/bin/activate"
  uv pip install aiperf
  deactivate
fi

# ---- EVAL_CMD will be appended below ----
BASH

  printf '\n%s\n' "$EVAL_CMD" >> "$RUN_SCRIPT_HOST"
  chmod +x "$RUN_SCRIPT_HOST"

  if command -v dos2unix >/dev/null 2>&1; then
    dos2unix -q "$RUN_SCRIPT_HOST" || true
  else
    sed -i 's/\r$//' "$RUN_SCRIPT_HOST" || true
  fi

  local docker_env=()
  [[ -n "$HF_TOKEN" ]] && docker_env+=(-e "HF_TOKEN=$HF_TOKEN")
  docker_env+=(-e "AICONF_SRC=$AICONF_SRC_CONTAINER")
  docker_env+=(-e "VENV_PATH=$VENV_PATH")

  log "Starting container: $CONTAINER_NAME"
  docker run --rm --gpus all \
    --name "$CONTAINER_NAME" \
    --ipc=host --ulimit memlock=-1 \
    --network host \
    -e NVIDIA_VISIBLE_DEVICES=all \
    "${docker_env[@]}" \
    -v "$MODEL_LOCAL_DIR":"$CONTAINER_MODEL_PATH":rw \
    -v "$SAVE_DIR_HOST":"$CONTAINER_SAVE_DIR":rw \
    -v "$SRC_ROOT_HOST":"$AICONF_SRC_CONTAINER":rw \
    "$DYNAMO_IMAGE" \
    bash -lc "/bin/bash '$RUN_SCRIPT_CONT'"
}



# main
if [[ "${IN_CONTAINER,,}" == "true" ]]; then
  run_inside_container
else
  run_with_docker
fi

log "Done."
