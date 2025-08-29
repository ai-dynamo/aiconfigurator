
# Automation: build + deploy + evaluate

This folder contains `launch_eval.sh`, a script that:

* downloads model weights (optional),
* ensures a suitable **Dynamo** container image (based on the config.env),
* launch **etcd** + **NATS**,
* enters the container and runs `aiconfigurator eval`.

> Run:
>
> ```bash
> ./aiconfigurator/tools/automation/launch_eval.sh /path/to/config.env
> ```

## 1) Running **inside** a container

**Assumptions**

* **etcd** and **NATS** are accessible.
* The container already has **aiconfigurator** and **dynamo** installed.
* The **dynamo** repo is mounted at `/workspace/`.

**How to use**

1. Create a `config.env` with `IN_CONTAINER=true`.
2. Make sure the model path in the config exists *inside* the container. If it does not exist, the script will fetch the corresponding model.
3. Run the script; it will directly call `aiconfigurator eval`.

Example `config.env` (minimum):

```bash
# --- Required paths ---
MODEL_LOCAL_DIR=/raid/hub/qwen3-32b-fp8
MODEL_HF_REPO=Qwen/Qwen3-32B-FP8

# --- Deployment knobs ---
SYSTEM=h200_sxm
MODEL=QWEN3_32B
VERSION=1.0.0rc3
GENERATED_CONFIG_VERSION=1.0.0rc6
ISL=5000
OSL=1000
TTFT=1000
TPOT=10
TOTAL_GPUS=8
HEAD_NODE_IP=0.0.0.0
PREFILL_FREE_GPU_MEM_FRAC=0.9
FREE_GPU_MEM_FRAC=0.7
DECODE_FREE_GPU_MEM_FRAC=0.5
PORT=8000
MODE=disagg

# --- Container image ---
DYNAMO_IMAGE=dynamo:0.5.0-trtllm-1.0.0rc6
TRTLLM_PIP=tensorrt-llm==1.0.0rc6

# --- Dynamo repo settings ---
# DYNAMO_DIR=/path/to/existing/dynamo  # Optional
DYNAMO_BRANCH=main
DYNAMO_GIT=https://github.com/ai-dynamo/dynamo

# --- Service naming ---
SERVED_MODEL_NAME=Qwen3/Qwen3-32B-FP8
CONTAINER_NAME=dynamo-single-node

# --- Benchmarking settings ---
BENCHMARK_CONCURRENCY=auto   # can be auto or cc list likes BENCHMARK_CONCURRENCY="1 4 8 12 16 20"
#BENCHMARK_CONCURRENCY="1 4 8"

# --- Optional: download model if not present ---
ENABLE_MODEL_DOWNLOAD=true

# --- Optional: run-time save dir mapping for host ---
# SAVE_DIR_HOST=/workspace/aiconf_save

# --- Optional: already inside the image ---
IN_CONTAINER=false
```

The script will skip image build and compose, and directly run:

```
aiconfigurator eval ...
```

---

## 2) Running **outside** a container

**What the script does**

1. Checks whether the target **Dynamo** container image exists.

   * If not, it clones/updates the **dynamo** repo and **builds** the image:

     ```
     ./container/build.sh --framework TRTLLM \
       --tensorrtllm-pip-wheel <TRTLLM_PIP> \
       --tag <DYNAMO_IMAGE>
     ```
2. Launch **etcd** and **NATS** using:

   ```
   docker compose -f deploy/docker-compose.yml up -d
   ```

3. Starts the container with host networking and GPUs.
4. Inside the container:

   * If `aiconfigurator` is missing, it **mounts the project source** to `/opt/aiconfigurator-src` and runs:

     ```
     python3 -m pip install -e /opt/aiconfigurator-src
     ```
   * Then it executes `aiconfigurator eval`.

Example `config.env` (minimum):

```bash
# --- Required paths ---
MODEL_LOCAL_DIR=/raid/hub/qwen3-32b-fp8
MODEL_HF_REPO=Qwen/Qwen3-32B-FP8

# --- Deployment knobs ---
SYSTEM=h200_sxm
MODEL=QWEN3_32B
VERSION=1.0.0rc3
GENERATED_CONFIG_VERSION=1.0.0rc6
ISL=5000
OSL=1000
TTFT=1000
TPOT=10
TOTAL_GPUS=8
HEAD_NODE_IP=0.0.0.0
PREFILL_FREE_GPU_MEM_FRAC=0.9
FREE_GPU_MEM_FRAC=0.7
DECODE_FREE_GPU_MEM_FRAC=0.5
PORT=8000
MODE=disagg

# --- Container image ---
DYNAMO_IMAGE=dynamo:0.5.0-trtllm-1.0.0rc6
TRTLLM_PIP=tensorrt-llm==1.0.0rc6

# --- Dynamo repo settings ---
# DYNAMO_DIR=/path/to/existing/dynamo  # Optional
DYNAMO_BRANCH=main
DYNAMO_GIT=https://github.com/ai-dynamo/dynamo

# --- Service naming ---
SERVED_MODEL_NAME=Qwen3/Qwen3-32B-FP8
CONTAINER_NAME=dynamo-single-node

# --- Benchmarking settings ---
BENCHMARK_CONCURRENCY=auto   # can be auto or cc list likes BENCHMARK_CONCURRENCY="1 4 8 12 16 20"
#BENCHMARK_CONCURRENCY="1 4 8"

# --- Optional: download model if not present ---
ENABLE_MODEL_DOWNLOAD=true

# --- Optional: run-time save dir mapping for host ---
# SAVE_DIR_HOST=/workspace/aiconf_save

# --- Optional: already inside the image ---
IN_CONTAINER=false
```

---

## Outputs

* **Service logs** (from `aiconfigurator eval`’s service launcher):
  `/<save_dir>/log/<run_name>_<mode>_p<port>.log` (inside the container mount).
* **Evaluation results**:
  `/<save_dir>/eval_runs/<run_name>/...` (bench JSON/CSV, Pareto plot, GPU stats, etc.).

---

## Notes

* If you use **Hugging Face** models, export `HF_TOKEN` before running the script so it can download weights.
* The script mounts:

  * **model directory** to `/workspace/model_hub/<name>`,
  * **save directory** to `/workspace/aiconf_save`,
  * **project root** to `/opt/aiconfigurator-src` for editable install (only if needed).
