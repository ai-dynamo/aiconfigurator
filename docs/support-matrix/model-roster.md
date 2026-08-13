# Support-matrix model roster

The default support-matrix generation roster is curated separately from the
model configurations bundled with AIConfigurator.

A bundled model remains available for explicit SDK and CLI use even after it
is retired from default matrix generation. This keeps historical workflows and
offline model loading working without requiring every superseded release to
occupy the full model/system/backend/version cross-product.

## Inclusion policy

Default matrix entries should represent at least one of the following:

- a current flagship model or checkpoint;
- a distinct architecture or operation pipeline;
- a precision variant with materially different runtime or performance-data
  requirements; or
- a compatibility case that protects an actively supported backend path.

Before adding a model, confirm that its runtime path is viable on at least one
matrix system/backend/version combination. Deterministic unsupported paths
must remain explicitly classified rather than reported as passing.

## Multimodal encoder coverage

The support matrix automatically exercises a checkpoint's vision encoder when
the checkpoint contains a non-empty `vision_config` and AIC implements that
encoder. The canonical workload is **one 1024 x 1024 image per request**. The
same image-bearing run covers the language backbone; multimodal checkpoints do
not receive a second, redundant text-only run.

An encoder-supported PASS means that the agg or disagg run used the canonical
image workload and produced strictly positive encoder latency and encoder
memory for every result row. `ImageHeight`, `ImageWidth`, and `NumImages` in the
generated CSV, plus the matching replay-command arguments, record that workload.

If a checkpoint declares `vision_config` but AIC cannot normalize and build its
encoder operations, the row fails with an `ENCODER_UNSUPPORTED` reason. It must
not inherit PASS from a successful text-backbone-only estimate. Text-only
checkpoints keep their existing workload and leave the image metadata empty.

## Retired from default generation

The following bundled configs remain usable explicitly but are superseded in
the default matrix:

- GLM-5 and GLM-5.1 in BF16, FP8, and NVFP4; GLM-5.2 remains.
- MiniMax-M2.5 in BF16 and NVFP4; MiniMax-M2.7 and MiniMax-M3 remain.
- Llama-3.3-Nemotron-Super-49B-v1 and Nemotron-H-56B-Base-8K;
  Nemotron-3 remains.

The source of truth is `SupportMatrixHFModels` in
`aiconfigurator_core.sdk.common`. `DefaultHFModels` remains the bundled config
inventory for compatibility and offline loading.
