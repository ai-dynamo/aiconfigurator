#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Prepare a disposable Brev GPU VM for unattended Codex runs against NVIDIA's
# OpenAI-compatible inference endpoint. This stores credentials in a local
# user-only env file; do not use it on shared hosts.

MODEL="${MODEL:-openai/openai/gpt-5.3-codex}"
BASE_URL="${NVIDIA_INFERENCE_BASE_URL:-https://inference-api.nvidia.com/v1}"
ENV_FILE="${NVIDIA_INFERENCE_ENV_FILE:-$HOME/.config/nvidia/api-env}"
BIN_DIR="${BIN_DIR:-$HOME/.local/bin}"
NVM_VERSION="${NVM_VERSION:-v0.39.7}"
NODE_VERSION="${NODE_VERSION:-20}"

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

need_cmd curl
need_cmd python3

echo "Setting up Brev VM for Codex YOLO mode"
echo "Model: $MODEL"
echo "Base URL: $BASE_URL"

if [ -z "${NVIDIA_INFERENCE_API_KEY:-}" ]; then
  read -rsp "Paste NVIDIA inference sk key: " NVIDIA_INFERENCE_API_KEY
  echo
fi

mkdir -p "$(dirname "$ENV_FILE")" "$BIN_DIR"
chmod 700 "$(dirname "$ENV_FILE")"

cat >"$ENV_FILE" <<ENVEOF
export NVIDIA_INFERENCE_BASE_URL="$BASE_URL"
export NVIDIA_INFERENCE_API_KEY="$NVIDIA_INFERENCE_API_KEY"
export OPENAI_BASE_URL="\$NVIDIA_INFERENCE_BASE_URL"
export OPENAI_API_KEY="\$NVIDIA_INFERENCE_API_KEY"
export CODEX_MODEL="$MODEL"
export CODEX_NODE_VERSION="$NODE_VERSION"
ENVEOF

chmod 600 "$ENV_FILE"

for rc in "$HOME/.bashrc" "$HOME/.profile" "$HOME/.zshrc"; do
  touch "$rc"
  grep -qF "$ENV_FILE" "$rc" || cat >>"$rc" <<RCEOF

# NVIDIA inference API
[ -f "$ENV_FILE" ] && source "$ENV_FILE"

# User-local binaries
export PATH="\$HOME/.local/bin:\$PATH"

# nvm / Node.js
export NVM_DIR="\$HOME/.nvm"
[ -s "\$NVM_DIR/nvm.sh" ] && . "\$NVM_DIR/nvm.sh"
RCEOF
done

# shellcheck disable=SC1090
source "$ENV_FILE"
export PATH="$BIN_DIR:$PATH"

echo "Testing NVIDIA inference API..."
curl -fsS "$NVIDIA_INFERENCE_BASE_URL/models" \
  -H "Authorization: Bearer $NVIDIA_INFERENCE_API_KEY" \
  -o /tmp/nvidia-models.json

echo "Models endpoint OK. First few models:"
python3 - <<'PY'
import json
with open("/tmp/nvidia-models.json") as f:
    data = json.load(f)
for m in data.get("data", [])[:8]:
    print(" -", m.get("id"))
PY

export NVM_DIR="$HOME/.nvm"

if [ ! -s "$NVM_DIR/nvm.sh" ]; then
  echo "Installing nvm..."
  curl -fsSL "https://raw.githubusercontent.com/nvm-sh/nvm/${NVM_VERSION}/install.sh" | bash
fi

# shellcheck disable=SC1091
. "$NVM_DIR/nvm.sh"

echo "Installing/using Node.js $NODE_VERSION via nvm..."
nvm install "$NODE_VERSION"
nvm use "$NODE_VERSION"
nvm alias default "$NODE_VERSION"

echo "Node version: $(node --version)"
echo "npm version: $(npm --version)"

if ! command -v codex >/dev/null 2>&1; then
  echo "Installing Codex CLI with user-local npm..."
  npm install -g @openai/codex
else
  echo "Codex CLI already installed: $(codex --version || true)"
fi

echo "Codex version:"
codex --version || true

cat >"$BIN_DIR/codex-yolo" <<SH
#!/usr/bin/env bash
set -euo pipefail

[ -f "$ENV_FILE" ] && source "$ENV_FILE"

export PATH="\$HOME/.local/bin:\$PATH"
export NVM_DIR="\$HOME/.nvm"
[ -s "\$NVM_DIR/nvm.sh" ] && . "\$NVM_DIR/nvm.sh"
nvm use "\${CODEX_NODE_VERSION:-$NODE_VERSION}" >/dev/null 2>&1 || true

exec codex -m "\${CODEX_MODEL:-$MODEL}" \
  --dangerously-bypass-approvals-and-sandbox "\$@"
SH

chmod +x "$BIN_DIR/codex-yolo"

echo
echo "Done."
echo
echo "For this current shell, run:"
echo "  source ~/.bashrc"
echo
echo "Then start YOLO mode with:"
echo "  codex-yolo"
echo
echo "Direct command:"
echo "  codex -m \"$MODEL\" --dangerously-bypass-approvals-and-sandbox"
