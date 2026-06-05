# Layerwise Collector

vLLM layerwise/FPM collection tools for comparing one-layer GPU attribution
against Dynamo forward-pass metrics.

Included paths:

- `vllm/`: vLLM layerwise collector, layer-skip patch, and step marker.
- `fpm_ground_truth/`: Dynamo FPM ground-truth collection helpers.
- `common/`: shared config patching, nsys parsers, and random prompt-token helpers.

The SGLang and TensorRT-LLM prototype collectors are intentionally not included
in this AIC branch.

Current validation data is stored under
`src/aiconfigurator/systems/data/b300_sxm/vllm/0.20.1/layerwise_perf.csv`.
Decode uses one-layer span data. Context currently uses 16-layer measurements
normalized per layer because one-layer context underpredicts standard-random
FPM at long sequence lengths.
