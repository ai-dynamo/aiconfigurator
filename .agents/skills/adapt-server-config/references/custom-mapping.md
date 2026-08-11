# Confirmed custom mapping

Use this workflow only when the source is neither an InferenceX DB record, a standard DynamoGraphDeployment, nor a concrete `dynamo-ci` SGLang benchmark recipe.

## Required confirmation table

Show one row per canonical field with these columns:

| Canonical group | Required values |
| --- | --- |
| Model | model path; speculative depth and accepted tokens when enabled |
| Quantization | GEMM, KV cache, FMHA, MoE, and communication overrides when explicit |
| Backend | vLLM, SGLang, or TRT-LLM; optional pinned version; database mode |
| Systems | prefill system and decode system for disaggregation |
| Workload | ISL, OSL, concurrency, prefix, and optional image shape |
| Topology | agg or disagg; replicas, GPUs per replica, batch, TP, PP, attention DP, MoE TP/EP per role |
| Runtime | memory fraction, max sequence length, system paths, and engine-step backend when explicit |
| Provenance | source reference, source identifiers, and every assumption |

For each value, label it `explicit`, `inferred`, `default`, or `missing`, and cite the source path or reasoning. Ask the user to confirm all inferred/default values and supply missing required values.

## Safety rules

- Preserve source operating-point order and create one result per point.
- Reject conflicting values instead of choosing one.
- Never execute shell, templates, or environment-producing commands.
- Accept only literal environment substitutions.
- Reject agg/encode/decode three-way splits, AFD, heterogeneous systems, and other topologies outside agg or P/D disaggregation.
- Require accepted speculative tokens whenever speculative decoding is active.
- Record confirmed assumptions in `provenance.assumptions`.

After confirmation, construct canonical JSON only. Let the packaged schema and Python model perform final validation.
