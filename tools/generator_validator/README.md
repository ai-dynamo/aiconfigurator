# Generator Validator

The generator validator checks that generated engine configs or CLI args are accepted by the backend runtime version. It is intended for quick sanity checks after running the generator.

## What it validates
- TRT-LLM: `agg_config.yaml`, `prefill_config.yaml`, `decode_config.yaml` by loading `tensorrt_llm.llmapi.llm_args.TorchLlmArgs`.
- vLLM: `k8s_deploy.yaml` by parsing CLI args with `vllm.engine.arg_utils.EngineArgs`.
- SGLang: `k8s_deploy.yaml` by parsing CLI args with `sglang.srt.server_args.ServerArgs`.

## Usage (run inside the matching runtime image)
TRT-LLM runtime image:
```
python -m aiconfigurator/tools/generator_validator/validator.py \
  --backend trtllm \
  --path /path/to/results
```

vLLM runtime image:
```
python -m aiconfigurator/tools/generator_validator/validator.py \
  --backend vllm \
  --path /path/to/results
```

SGLang runtime image:
```
python -m aiconfigurator/tools/generator_validator/validator.py \
  --backend sglang \
  --path /path/to/results
```

## `--path` meaning (file or directory)
- File: point directly to a single engine config YAML (TRT-LLM) or `k8s_deploy.yaml` (vLLM/SGLang).
- Directory: point to a generator results root with the expected layout:
  - TRT-LLM: `agg/top1/agg_config.yaml` and `disagg/top1/prefill_config.yaml`, `disagg/top1/decode_config.yaml`
  - vLLM / SGLang: `agg/top1/k8s_deploy.yaml` and `disagg/top1/k8s_deploy.yaml`

## Expected result layout
- TRT-LLM: `agg/top1/agg_config.yaml`, `disagg/top1/prefill_config.yaml`, `disagg/top1/decode_config.yaml`
- vLLM / SGLang: `agg/top1/k8s_deploy.yaml`, `disagg/top1/k8s_deploy.yaml`
