# Support-matrix model roster

The default support-matrix generation roster is curated separately from the
model configurations bundled with AIConfigurator.

A bundled model remains available for explicit SDK and CLI use even after it
is retired from default matrix generation. This keeps historical workflows and
offline model loading working without requiring every superseded release to
occupy the full model/system/backend/version cross product.

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
