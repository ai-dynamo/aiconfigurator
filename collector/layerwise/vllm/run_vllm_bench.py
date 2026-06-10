"""Path-A vLLM layerwise bench driver (vLLM 0.19.0 v1).

Wraps `vllm.benchmarks.latency` with three monkey-patches:
  1. `vllm_layer_skip_patch` — identity forward on non-target DecoderLayers
  2. `vllm_step_marker`      — NVTX outer range at target iteration numbers
  3. `config_patch`          — HF config.json dotted-key override + aux-file pull

Canonical run (GLM-5-FP8 decode, bs=128, past_kv sweep):
  nsys profile --cuda-graph-trace=node -o nsys_sweep \\
    python3 run_vllm_bench.py \\
      --model zai-org/GLM-5-FP8 \\
      --config-overrides '{"num_hidden_layers": 4}' \\
      --no-async-scheduling --kv-cache-dtype fp8 \\
      --compilation-config '{"mode": 0, "cudagraph_mode": "FULL"}' \\
      --batch-size 128 --input-len 1 --output-len 8192 \\
      --max-model-len 8256 --num-iters-warmup 0 --num-iters 1

See ../README.md for the three-framework alignment rule and parser usage.
"""
import argparse
import importlib
import json
import logging
import multiprocessing as mp
import os
import sys

# Make sibling dirs importable.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)  # vllm_layer_skip_patch, vllm_step_marker
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "common"))  # config_patch

# Run the engine in-process and import patches at driver top level so forked
# children inherit sys.modules.
print("[run_vllm_bench] fork mode: V1_MP=0 + fork start method")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
importlib.import_module("vllm_layer_skip_patch")
importlib.import_module("vllm_step_marker")

try:
    mp.set_start_method("fork", force=True)
except RuntimeError:
    pass

from config_patch import patch_model_path


def _pop_arg(argv, flag, has_value=True):
    """Remove `--flag [value]` from argv and return value (or True for bool flags)."""
    i = 0
    val = None
    while i < len(argv):
        if argv[i] == flag:
            if has_value and i + 1 < len(argv):
                val = argv[i + 1]
                del argv[i : i + 2]
            else:
                val = True
                del argv[i]
            return val
        i += 1
    return val


def _maybe_patch_model_path(argv):
    """Handle our extra --config-overrides flag; rewrite --model if provided."""
    overrides_json = _pop_arg(argv, "--config-overrides", has_value=True)
    if overrides_json is None:
        return argv

    model = None
    for i, a in enumerate(argv):
        if a == "--model" and i + 1 < len(argv):
            model = argv[i + 1]
            idx = i + 1
            break
    if model is None:
        logging.warning("[run_vllm_bench] --config-overrides given but no --model; skipping")
        return argv

    overrides = json.loads(overrides_json)
    # Known-bad model_type strings that HF transformers doesn't parse.
    # Rewrite to a compatible model_type while keeping `architectures` intact
    # so vLLM's registry still dispatches to the native subclass.
    #   - glm_moe_dsa (GLM-5 DSA) → deepseek_v3
    # Qwen3.5-122B-A10B (model_type=qwen3_5_moe) is parsed natively; do NOT
    # rewrite. Any other model_type the user cares about should be added here.
    rewrites = {"glm_moe_dsa": "deepseek_v3"}
    local = patch_model_path(
        model, overrides,
        strip_auto_map=True,
        model_type_rewrites=rewrites,
    )
    print(f"[config-patch] model {model} -> {local}")
    argv[idx] = local
    return argv


def main():
    sys.argv = [sys.argv[0]] + _maybe_patch_model_path(sys.argv[1:])

    # Defer vllm imports until patches are installed and argv rewritten.
    from vllm.benchmarks.latency import add_cli_args
    from vllm.benchmarks.latency import main as latency_main

    parser = argparse.ArgumentParser()
    add_cli_args(parser)
    parser.set_defaults(
        load_format="dummy",
        trust_remote_code=True,
        enable_layerwise_nvtx_tracing=True,
        skip_tokenizer_init=True,  # bench uses dummy token_ids; tokenizer not needed
        disable_detokenize=True,
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    latency_main(args)


if __name__ == "__main__":
    main()
